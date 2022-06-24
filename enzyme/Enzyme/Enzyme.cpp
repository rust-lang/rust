//===- Enzyme.cpp - Automatic Differentiation Transformation Pass  -------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains Enzyme, a transformation pass that takes replaces calls
// to function calls to *__enzyme_autodiff* with a call to the derivative of
// the function passed as the first argument.
//
//===----------------------------------------------------------------------===//
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Passes/PassBuilder.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Transforms/Utils/Cloning.h"
#if LLVM_VERSION_MAJOR >= 11
#include "llvm/Analysis/InlineAdvisor.h"
#endif
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "ActivityAnalysis.h"
#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "Utils.h"

#include "llvm/Transforms/Utils.h"

#if LLVM_VERSION_MAJOR >= 13
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/OpenMPOpt.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#endif

#include "CApi.h"
using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-enzyme-intrinsic"

llvm::cl::opt<bool>
    EnzymePostOpt("enzyme-postopt", cl::init(false), cl::Hidden,
                  cl::desc("Run enzymepostprocessing optimizations"));

llvm::cl::opt<bool> EnzymeAttributor("enzyme-attributor", cl::init(false),
                                     cl::Hidden,
                                     cl::desc("Run attributor post Enzyme"));

llvm::cl::opt<bool> EnzymeOMPOpt("enzyme-omp-opt", cl::init(false), cl::Hidden,
                                 cl::desc("Whether to enable openmp opt"));

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#endif
namespace {

template <const char *handlername, DerivativeMode Mode, int numargs>
static void
handleCustomDerivative(llvm::Module &M, llvm::GlobalVariable &g,
                       SmallVectorImpl<GlobalVariable *> &globalsToErase) {
  if (g.hasInitializer()) {
    if (auto CA = dyn_cast<ConstantAggregate>(g.getInitializer())) {
      if (CA->getNumOperands() != numargs) {
        llvm::errs() << M << "\n";
        llvm::errs() << "Use of " << handlername
                     << " must be a "
                        "constant of size "
                     << numargs << " " << g << "\n";
        llvm_unreachable(handlername);
      } else {
        Function *Fs[numargs];
        for (size_t i = 0; i < numargs; i++) {
          Value *V = CA->getOperand(i);
          while (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
          }
          if (auto CA = dyn_cast<ConstantAggregate>(V))
            V = CA->getOperand(0);
          while (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
          }
          if (auto F = dyn_cast<Function>(V)) {
            Fs[i] = F;
          } else {
            llvm::errs() << M << "\n";
            llvm::errs() << "Param of " << handlername
                         << " must be a "
                            "function"
                         << g << "\n"
                         << *V << "\n";
            llvm_unreachable(handlername);
          }
        }

        if (Mode == DerivativeMode::ReverseModeGradient) {
          assert(numargs == 3);
          Fs[0]->setMetadata(
              "enzyme_augment",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[1])}));
          Fs[0]->setMetadata(
              "enzyme_gradient",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[2])}));
        } else if (Mode == DerivativeMode::ForwardMode) {
          assert(numargs == 2);
          Fs[0]->setMetadata(
              "enzyme_derivative",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[1])}));
        } else if (Mode == DerivativeMode::ForwardModeSplit) {
          assert(numargs == 3);
          Fs[0]->setMetadata(
              "enzyme_augment",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[1])}));
          Fs[0]->setMetadata(
              "enzyme_splitderivative",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[2])}));
        } else
          assert("Unknown mode");
      }
    } else {
      llvm::errs() << M << "\n";
      llvm::errs() << "Use of " << handlername
                   << " must be a "
                      "constant aggregate "
                   << g << "\n";
      llvm_unreachable(handlername);
    }
  } else {
    llvm::errs() << M << "\n";
    llvm::errs() << "Use of " << handlername
                 << " must be a "
                    "constant array of size "
                 << numargs << " " << g << "\n";
    llvm_unreachable(handlername);
  }
  globalsToErase.push_back(&g);
}

static void
handleInactiveFunction(llvm::Module &M, llvm::GlobalVariable &g,
                       SmallVectorImpl<GlobalVariable *> &globalsToErase) {
  if (g.hasInitializer()) {
    Value *V = g.getInitializer();
    while (auto CE = dyn_cast<ConstantExpr>(V)) {
      V = CE->getOperand(0);
    }
    if (auto CA = dyn_cast<ConstantAggregate>(V))
      V = CA->getOperand(0);
    while (auto CE = dyn_cast<ConstantExpr>(V)) {
      V = CE->getOperand(0);
    }
    if (auto F = dyn_cast<Function>(V)) {
      F->addAttribute(AttributeList::FunctionIndex,
                      Attribute::get(g.getContext(), "enzyme_inactive"));
    } else {
      llvm::errs() << M << "\n";
      llvm::errs() << "Param of __enzyme_inactivefn must be a "
                      "function"
                   << g << "\n"
                   << *V << "\n";
      llvm_unreachable("__enzyme_inactivefn");
    }
  } else {
    llvm::errs() << M << "\n";
    llvm::errs() << "Use of __enzyme_inactivefn must be a "
                    "constant function "
                 << g << "\n";
    llvm_unreachable("__enzyme_register_gradient");
  }
  globalsToErase.push_back(&g);
}

static void handleKnownFunctions(llvm::Function &F) {
  if (F.getName() == "MPI_Irecv" || F.getName() == "PMPI_Irecv") {
    F.addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    F.addParamAttr(0, Attribute::WriteOnly);
    if (F.getFunctionType()->getParamType(2)->isPointerTy()) {
      F.addParamAttr(2, Attribute::NoCapture);
      F.addParamAttr(2, Attribute::WriteOnly);
    }
    F.addParamAttr(6, Attribute::WriteOnly);
  }
  if (F.getName() == "MPI_Isend" || F.getName() == "PMPI_Isend") {
    F.addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    F.addParamAttr(0, Attribute::ReadOnly);
    if (F.getFunctionType()->getParamType(2)->isPointerTy()) {
      F.addParamAttr(2, Attribute::NoCapture);
      F.addParamAttr(2, Attribute::ReadOnly);
    }
    F.addParamAttr(6, Attribute::WriteOnly);
  }
  if (F.getName() == "MPI_Comm_rank" || F.getName() == "PMPI_Comm_rank" ||
      F.getName() == "MPI_Comm_size" || F.getName() == "PMPI_Comm_size") {
    F.addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    if (F.getFunctionType()->getParamType(0)->isPointerTy()) {
      F.addParamAttr(0, Attribute::NoCapture);
      F.addParamAttr(0, Attribute::ReadOnly);
    }
    if (F.getFunctionType()->getParamType(1)->isPointerTy()) {
      F.addParamAttr(1, Attribute::WriteOnly);
      F.addParamAttr(1, Attribute::NoCapture);
    }
  }
  if (F.getName() == "MPI_Wait" || F.getName() == "PMPI_Wait") {
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    F.addParamAttr(0, Attribute::NoCapture);
    F.addParamAttr(1, Attribute::WriteOnly);
    F.addParamAttr(1, Attribute::NoCapture);
  }
  if (F.getName() == "MPI_Waitall" || F.getName() == "PMPI_Waitall") {
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    F.addParamAttr(1, Attribute::NoCapture);
    F.addParamAttr(2, Attribute::WriteOnly);
    F.addParamAttr(2, Attribute::NoCapture);
  }
  if (F.getName() == "omp_get_max_threads" ||
      F.getName() == "omp_get_thread_num") {
    F.addFnAttr(Attribute::ReadOnly);
    F.addFnAttr(Attribute::InaccessibleMemOnly);
  }
  if (F.getName() == "frexp" || F.getName() == "frexpf" ||
      F.getName() == "frexpl") {
    F.addFnAttr(Attribute::ArgMemOnly);
    F.addParamAttr(1, Attribute::WriteOnly);
  }
  if (F.getName() == "__fd_sincos_1" || F.getName() == "__fd_cos_1" ||
      F.getName() == "__mth_i_ipowi") {
    F.addFnAttr(Attribute::ReadNone);
  }
}

static void handleAnnotations(llvm::Function &F) {
  if (F.getName().contains("__enzyme_float") ||
      F.getName().contains("__enzyme_double") ||
      F.getName().contains("__enzyme_integer") ||
      F.getName().contains("__enzyme_pointer") ||
      F.getName().contains("__enzyme_virtualreverse")) {
    F.addFnAttr(Attribute::ReadNone);
    for (auto &arg : F.args()) {
      if (arg.getType()->isPointerTy()) {
        arg.addAttr(Attribute::ReadNone);
        arg.addAttr(Attribute::NoCapture);
      }
    }
  }
}

static Value *
castToDiffeFunctionArgType(IRBuilder<> &Builder, llvm::CallInst *CI,
                           llvm::FunctionType *FT, llvm::Type *destType,
                           unsigned int i, DerivativeMode mode,
                           llvm::Value *value, unsigned int truei) {
  auto res = value;
  if (auto ptr = dyn_cast<PointerType>(res->getType())) {
    if (auto PT = dyn_cast<PointerType>(destType)) {
      if (ptr->getAddressSpace() != PT->getAddressSpace()) {
        res = Builder.CreateAddrSpaceCast(
            res, PointerType::get(ptr->getPointerElementType(),
                                  PT->getAddressSpace()));
        assert(value);
        assert(destType);
        assert(FT);
        llvm::errs() << "Warning cast(2) __enzyme_autodiff argument " << i
                     << " " << *res << "|" << *res->getType() << " to argument "
                     << truei << " " << *destType << "\n"
                     << "orig: " << *FT << "\n";
        return res;
      }
    }
  }

  if (!res->getType()->canLosslesslyBitCastTo(destType)) {
    assert(value);
    assert(value->getType());
    assert(destType);
    assert(FT);
    auto loc = CI->getDebugLoc();
    if (auto arg = dyn_cast<Instruction>(res)) {
      loc = arg->getDebugLoc();
    }
    EmitFailure("IllegalArgCast", loc, CI,
                "Cannot cast __enzyme_autodiff shadow argument ", i, ", found ",
                *res, ", type ", *res->getType(), " - to arg ", truei, " ",
                *destType);
    return nullptr;
  }
  return Builder.CreateBitCast(value, destType);
}

static Optional<StringRef> getMetadataName(llvm::Value *res) {
  if (auto av = dyn_cast<MetadataAsValue>(res)) {
    return cast<MDString>(av->getMetadata())->getString();
  } else if ((isa<LoadInst>(res) || isa<CastInst>(res)) &&
             isa<GlobalVariable>(cast<Instruction>(res)->getOperand(0))) {
    GlobalVariable *gv =
        cast<GlobalVariable>(cast<Instruction>(res)->getOperand(0));
    return gv->getName();
  } else if (isa<LoadInst>(res) &&
             isa<ConstantExpr>(cast<LoadInst>(res)->getOperand(0)) &&
             cast<ConstantExpr>(cast<LoadInst>(res)->getOperand(0))->isCast() &&
             isa<GlobalVariable>(
                 cast<ConstantExpr>(cast<LoadInst>(res)->getOperand(0))
                     ->getOperand(0))) {
    auto gv = cast<GlobalVariable>(
        cast<ConstantExpr>(cast<LoadInst>(res)->getOperand(0))->getOperand(0));
    return gv->getName();
  } else if (auto gv = dyn_cast<GlobalVariable>(res)) {
    return gv->getName();
  } else if (isa<ConstantExpr>(res) && cast<ConstantExpr>(res)->isCast() &&
             isa<GlobalVariable>(cast<ConstantExpr>(res)->getOperand(0))) {
    auto gv = cast<GlobalVariable>(cast<ConstantExpr>(res)->getOperand(0));
    return gv->getName();
  } else if (isa<CastInst>(res) && cast<CastInst>(res) &&
             isa<AllocaInst>(cast<CastInst>(res)->getOperand(0))) {
    auto gv = cast<AllocaInst>(cast<CastInst>(res)->getOperand(0));
    return gv->getName();
  } else if (auto gv = dyn_cast<AllocaInst>(res)) {
    return gv->getName();
  } else {
    return Optional<StringRef>();
  }
}

class Enzyme : public ModulePass {
public:
  EnzymeLogic Logic;
  static char ID;
  Enzyme(bool PostOpt = false)
      : ModulePass(ID), Logic(PostOpt | EnzymePostOpt) {
    // initializeLowerAutodiffIntrinsicPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();

    // AU.addRequiredID(LCSSAID);

    // LoopInfo is required to ensure that all loops have preheaders
    // AU.addRequired<LoopInfoWrapperPass>();

    // AU.addRequiredID(llvm::LoopSimplifyID);//<LoopSimplifyWrapperPass>();
  }

  Optional<Function *> parseFunctionParameter(CallInst *CI) {
    Value *fn = CI->getArgOperand(0);

    // determine function to differentiate
    if (CI->hasStructRetAttr()) {
      fn = CI->getArgOperand(1);
    }

    while (auto ci = dyn_cast<CastInst>(fn)) {
      fn = ci->getOperand(0);
    }
    while (auto ci = dyn_cast<BlockAddress>(fn)) {
      fn = ci->getFunction();
    }
    while (auto ci = dyn_cast<ConstantExpr>(fn)) {
      fn = ci->getOperand(0);
    }
    if (!isa<Function>(fn)) {
      EmitFailure("NoFunctionToDifferentiate", CI->getDebugLoc(), CI,
                  "failed to find fn to differentiate", *CI, " - found - ",
                  *fn);
      return None;
    }
    if (cast<Function>(fn)->empty()) {
      EmitFailure("EmptyFunctionToDifferentiate", CI->getDebugLoc(), CI,
                  "failed to find fn to differentiate", *CI, " - found - ",
                  *fn);
      return None;
    }

    return cast<Function>(fn);
  }

  Optional<unsigned> parseWidthParameter(CallInst *CI) {
    unsigned width = 1;

#if LLVM_VERSION_MAJOR >= 14
    for (auto [i, found] = std::tuple{0u, false}; i < CI->arg_size(); ++i)
#else
    for (auto [i, found] = std::tuple{0u, false}; i < CI->getNumArgOperands();
         ++i)
#endif
    {
      Value *arg = CI->getArgOperand(i);

      if (auto MDName = getMetadataName(arg)) {
        if (*MDName == "enzyme_width") {
          if (found) {
            EmitFailure("IllegalVectorWidth", CI->getDebugLoc(), CI,
                        "vector width declared more than once",
                        *CI->getArgOperand(i), " in", *CI);
            return None;
          }

#if LLVM_VERSION_MAJOR >= 14
          if (i + 1 >= CI->arg_size())
#else
          if (i + 1 >= CI->getNumArgOperands())
#endif
          {
            EmitFailure("MissingVectorWidth", CI->getDebugLoc(), CI,
                        "constant integer followong enzyme_width is missing",
                        *CI->getArgOperand(i), " in", *CI);
            return None;
          }

          Value *width_arg = CI->getArgOperand(i + 1);
          if (auto cint = dyn_cast<ConstantInt>(width_arg)) {
            width = cint->getZExtValue();
            found = true;
          } else {
            EmitFailure("IllegalVectorWidth", CI->getDebugLoc(), CI,
                        "enzyme_width must be a constant integer",
                        *CI->getArgOperand(i), " in", *CI);
            return None;
          }

          if (!found) {
            EmitFailure("IllegalVectorWidth", CI->getDebugLoc(), CI,
                        "illegal enzyme vector argument width ",
                        *CI->getArgOperand(i), " in", *CI);
            return None;
          }
        }
      }
    }
    return width;
  }

  /// Return whether successful
  bool HandleAutoDiff(CallInst *CI, TargetLibraryInfo &TLI, DerivativeMode mode,
                      bool sizeOnly) {

    // determine function to differentiate
    Function *fn;
    auto parsedFunction = parseFunctionParameter(CI);
    if (parsedFunction.hasValue()) {
      fn = parsedFunction.getValue();
    } else {
      return false;
    }

    auto FT = fn->getFunctionType();
    assert(fn);

    IRBuilder<> Builder(CI);
    unsigned truei = 0;
    unsigned width = 1;
    std::map<unsigned, Value *> batchOffset;
    bool returnUsed =
        !fn->getReturnType()->isVoidTy() && !fn->getReturnType()->isEmptyTy();

    // find and handle enzyme_width
    auto parsedWidth = parseWidthParameter(CI);
    if (parsedWidth.hasValue()) {
      width = parsedWidth.getValue();
    } else {
      return false;
    }

    std::vector<DIFFE_TYPE> constants;
    SmallVector<Value *, 2> args;

    // handle different argument order for struct return.
    bool sret = CI->hasStructRetAttr() ||
                fn->hasParamAttribute(0, Attribute::StructRet);
    if (fn->hasParamAttribute(0, Attribute::StructRet)) {
      Type *fnsrety = cast<PointerType>(FT->getParamType(0));

      truei = 1;

      const DataLayout &DL = CI->getParent()->getModule()->getDataLayout();
      Type *Ty = fnsrety->getPointerElementType();
#if LLVM_VERSION_MAJOR >= 11
      AllocaInst *primal = new AllocaInst(Ty, DL.getAllocaAddrSpace(), nullptr,
                                          DL.getPrefTypeAlign(Ty));
#else
      AllocaInst *primal = new AllocaInst(Ty, DL.getAllocaAddrSpace(), nullptr);
#endif

      primal->insertBefore(CI);

      Value *shadow;
      switch (mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode: {
        Value *sretPt = CI->getArgOperand(0);
        if (width > 1) {
          PointerType *pty = cast<PointerType>(sretPt->getType());
          if (auto sty = dyn_cast<StructType>(pty->getPointerElementType())) {
            Value *acc = UndefValue::get(
                ArrayType::get(PointerType::get(sty->getElementType(0),
                                                pty->getAddressSpace()),
                               width));
            for (size_t i = 0; i < width; ++i) {
#if LLVM_VERSION_MAJOR > 7
              Value *elem = Builder.CreateStructGEP(
                  sretPt->getType()->getPointerElementType(), sretPt, i);
#else
              Value *elem = Builder.CreateStructGEP(sretPt, i);
#endif
              acc = Builder.CreateInsertValue(acc, elem, i);
            }
            shadow = acc;
          } else {
            EmitFailure(
                "IllegalReturnType", CI->getDebugLoc(), CI,
                "Return type of __enzyme_autodiff has to be a struct with",
                width, "elements of the same type.");
            return false;
          }
        } else {
          shadow = sretPt;
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:
      case DerivativeMode::ReverseModeCombined:
      case DerivativeMode::ReverseModeGradient: {
        shadow = CI->getArgOperand(1);
        sret = true;
        break;
      }
      }

      args.push_back(primal);
      args.push_back(shadow);
      constants.push_back(DIFFE_TYPE::DUP_ARG);
    }

    if (EnzymePrint)
      llvm::errs() << "prefn:\n" << *fn << "\n";

    auto Arch =
        llvm::Triple(
            CI->getParent()->getParent()->getParent()->getTargetTriple())
            .getArch();

    bool AtomicAdd = Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
                     Arch == Triple::amdgcn;

    bool freeMemory = true;

    DIFFE_TYPE retType = whatType(fn->getReturnType(), mode);

    bool differentialReturn = (mode == DerivativeMode::ReverseModeCombined ||
                               mode == DerivativeMode::ReverseModeGradient) &&
                              (retType == DIFFE_TYPE::OUT_DIFF);

    std::map<int, Type *> byVal;
    llvm::Value *differet = nullptr;
    llvm::Value *tape = nullptr;
    bool tapeIsPointer = false;
    int allocatedTapeSize = -1;

#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 1 + sret; i < CI->arg_size(); ++i)
#else
    for (unsigned i = 1 + sret; i < CI->getNumArgOperands(); ++i)
#endif
    {
      Value *res = CI->getArgOperand(i);
      if (truei >= FT->getNumParams()) {
        if (!isa<MetadataAsValue>(res) &&
            (mode == DerivativeMode::ReverseModeGradient ||
             mode == DerivativeMode::ForwardModeSplit)) {
          if (differentialReturn && differet == nullptr) {
            differet = res;
            if (CI->paramHasAttr(i, Attribute::ByVal)) {
#if LLVM_VERSION_MAJOR > 7
              differet = Builder.CreateLoad(
                  differet->getType()->getPointerElementType(), differet);
#else
              differet = Builder.CreateLoad(differet);
#endif
            }
            assert(differet->getType() == fn->getReturnType());
            continue;
          } else if (tape == nullptr) {
            tape = res;
            if (CI->paramHasAttr(i, Attribute::ByVal)) {
#if LLVM_VERSION_MAJOR > 7
              tape = Builder.CreateLoad(
                  tape->getType()->getPointerElementType(), tape);
#else
              tape = Builder.CreateLoad(tape);
#endif
            }
            continue;
          }
        }
        EmitFailure("TooManyArgs", CI->getDebugLoc(), CI,
                    "Had too many arguments to __enzyme_autodiff", *CI,
                    " - extra arg - ", *res);
        return false;
      }
      assert(truei < FT->getNumParams());
      auto PTy = FT->getParamType(truei);
      DIFFE_TYPE ty = whatType(PTy, mode);
      Optional<StringRef> metaString = getMetadataName(res);

      // handle metadata
      if (metaString && metaString.getValue().startswith("enzyme_")) {
        if (*metaString == "enzyme_dup") {
          ty = DIFFE_TYPE::DUP_ARG;
        } else if (*metaString == "enzyme_dupv") {
          ty = DIFFE_TYPE::DUP_ARG;
          ++i;
          Value *offset_arg = CI->getArgOperand(i);
          if (offset_arg->getType()->isIntegerTy()) {
            batchOffset[i + 1] = offset_arg;
          } else {
            EmitFailure("IllegalVectorOffset", CI->getDebugLoc(), CI,
                        "enzyme_batch must be followd by an integer "
                        "offset.",
                        *CI->getArgOperand(i), " in", *CI);
            return false;
          }
          continue;
        } else if (*metaString == "enzyme_dupnoneedv") {
          ty = DIFFE_TYPE::DUP_NONEED;
          ++i;
          Value *offset_arg = CI->getArgOperand(i);
          if (offset_arg->getType()->isIntegerTy()) {
            batchOffset[i + 1] = offset_arg;
          } else {
            EmitFailure("IllegalVectorOffset", CI->getDebugLoc(), CI,
                        "enzyme_batch must be followd by an integer "
                        "offset.",
                        *CI->getArgOperand(i), " in", *CI);
            return false;
          }
          continue;
        } else if (*metaString == "enzyme_dupnoneed") {
          ty = DIFFE_TYPE::DUP_NONEED;
        } else if (*metaString == "enzyme_dupnoneedv") {
          ty = DIFFE_TYPE::DUP_NONEED;
          ++i;
          Value *offset_arg = CI->getArgOperand(i);
          if (offset_arg->getType()->isIntegerTy()) {
            batchOffset[i + 1] = offset_arg;
          } else {
            EmitFailure("IllegalVectorOffset", CI->getDebugLoc(), CI,
                        "enzyme_batch must be followd by an integer "
                        "offset.",
                        *CI->getArgOperand(i), " in", *CI);
            return false;
          }
          continue;
        } else if (*metaString == "enzyme_out") {
          ty = DIFFE_TYPE::OUT_DIFF;
        } else if (*metaString == "enzyme_const") {
          ty = DIFFE_TYPE::CONSTANT;
        } else if (*metaString == "enzyme_noret") {
          returnUsed = false;
          continue;
        } else if (*metaString == "enzyme_allocated") {
          assert(!sizeOnly);
          ++i;
          if (!isa<ConstantInt>(CI->getArgOperand(i))) {
            EmitFailure("IllegalAllocatedSize", CI->getDebugLoc(), CI,
                        "illegal enzyme allocated size ", *CI->getArgOperand(i),
                        "in", *CI);
            return false;
          }
          allocatedTapeSize =
              cast<ConstantInt>(CI->getArgOperand(i))->getZExtValue();
          continue;
        } else if (*metaString == "enzyme_tape") {
          assert(!sizeOnly);
          ++i;
          tape = CI->getArgOperand(i);
          tapeIsPointer = true;
          continue;
        } else if (*metaString == "enzyme_nofree") {
          assert(!sizeOnly);
          freeMemory = false;
          continue;
        } else if (*metaString == "enzyme_width") {
          ++i;
          continue;
        } else {
          EmitFailure("IllegalDiffeType", CI->getDebugLoc(), CI,
                      "illegal enzyme metadata classification ", *CI,
                      *metaString);
          return false;
        }
        if (sizeOnly) {
          constants.push_back(ty);
          continue;
        }
        ++i;
        res = CI->getArgOperand(i);
      }

      constants.push_back(ty);

      assert(truei < FT->getNumParams());
      // cast primal
      if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
          if (auto PT = dyn_cast<PointerType>(PTy)) {
            if (ptr->getAddressSpace() != PT->getAddressSpace()) {
              res = Builder.CreateAddrSpaceCast(
                  res, PointerType::get(ptr->getPointerElementType(),
                                        PT->getAddressSpace()));
              assert(res);
              assert(PTy);
              assert(FT);
              llvm::errs() << "Warning cast(1) __enzyme_autodiff argument " << i
                           << " " << *res << "|" << *res->getType()
                           << " to argument " << truei << " " << *PTy << "\n"
                           << "orig: " << *FT << "\n";
            }
          }
        }
        if (res->getType()->canLosslesslyBitCastTo(PTy)) {
          res = Builder.CreateBitCast(res, PTy);
        }
        if (res->getType() != PTy && res->getType()->isIntegerTy() &&
            PTy->isIntegerTy(1)) {
          res = Builder.CreateTrunc(res, PTy);
        }
        if (res->getType() != PTy) {
          auto loc = CI->getDebugLoc();
          if (auto arg = dyn_cast<Instruction>(res)) {
            loc = arg->getDebugLoc();
          }
          EmitFailure("IllegalArgCast", loc, CI,
                      "Cannot cast __enzyme_autodiff primal argument ", i,
                      ", found ", *res, ", type ", *res->getType(),
                      " - to arg ", truei, " ", *PTy);
          return false;
        }
      }
#if LLVM_VERSION_MAJOR >= 9
      if (CI->isByValArgument(i)) {
        byVal[args.size()] = CI->getParamByValType(i);
      }
#endif
      args.push_back(res);
      if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
        ++i;

        Value *res = nullptr;
        bool batch = batchOffset.count(i - 1) != 0;

        for (unsigned v = 0; v < width; ++v) {
#if LLVM_VERSION_MAJOR >= 14
          if (i >= CI->arg_size())
#else
          if (i >= CI->getNumArgOperands())
#endif
          {
            EmitFailure("MissingArgShadow", CI->getDebugLoc(), CI,
                        "__enzyme_autodiff missing argument shadow at index ",
                        i, ", need shadow of type ", *PTy,
                        " to shadow primal argument ", *args.back(),
                        " at call ", *CI);
            return false;
          }

          // cast diffe
          Value *element = CI->getArgOperand(i);
          if (batch) {
            if (auto elementPtrTy = dyn_cast<PointerType>(element->getType())) {
              element = Builder.CreateBitCast(
                  element, PointerType::get(Type::getInt8Ty(CI->getContext()),
                                            elementPtrTy->getAddressSpace()));
#if LLVM_VERSION_MAJOR >= 7
              element = Builder.CreateGEP(
                  Type::getInt8Ty(CI->getContext()), element,
                  Builder.CreateMul(
                      batchOffset[i - 1],
                      ConstantInt::get(batchOffset[i - 1]->getType(), v)));
#else
              element = Builder.CreateGEP(
#if LLVM_VERSION_MAJOR >= 14
                  elementPtrTy,
#endif
                  element,
                  Builder.CreateMul(
                      batchOffset[i - 1],
                      ConstantInt::get(batchOffset[i - 1]->getType(), v)));
#endif
              element = Builder.CreateBitCast(element, elementPtrTy);
            } else {
              return false;
            }
          }
          if (PTy != element->getType()) {
            element = castToDiffeFunctionArgType(Builder, CI, FT, PTy, i, mode,
                                                 element, truei);
            if (!element) {
              return false;
            }
          }

          if (width > 1) {
            res =
                res ? Builder.CreateInsertValue(res, element, {v})
                    : Builder.CreateInsertValue(UndefValue::get(ArrayType::get(
                                                    element->getType(), width)),
                                                element, {v});

            if (v < width - 1 && !batch) {
              ++i;
            }

          } else {
            res = element;
          }
        }

        args.push_back(res);
      }

      ++truei;
    }

    std::map<Argument *, bool> volatile_args;
    FnTypeInfo type_args(fn);
    for (auto &a : type_args.Function->args()) {
      volatile_args[&a] = !(mode == DerivativeMode::ReverseModeCombined);
      TypeTree dt;
      if (a.getType()->isFPOrFPVectorTy()) {
        dt = ConcreteType(a.getType()->getScalarType());
      } else if (a.getType()->isPointerTy()) {
        auto et = a.getType()->getPointerElementType();
        if (et->isFPOrFPVectorTy()) {
          dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1);
        } else if (et->isPointerTy()) {
          dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1);
        }
        dt.insert({}, BaseType::Pointer);
      } else if (a.getType()->isIntOrIntVectorTy()) {
        dt = ConcreteType(BaseType::Integer);
      }
      type_args.Arguments.insert(
          std::pair<Argument *, TypeTree>(&a, dt.Only(-1)));
      // TODO note that here we do NOT propagate constants in type info (and
      // should consider whether we should)
      type_args.KnownValues.insert(
          std::pair<Argument *, std::set<int64_t>>(&a, {}));
    }

    TypeAnalysis TA(Logic.PPC.FAM);
    type_args = TA.analyzeFunction(type_args).getAnalyzedTypeInfo();

    // differentiate fn
    Function *newFunc = nullptr;
    Type *tapeType = nullptr;
    const AugmentedReturn *aug;
    switch (mode) {
    case DerivativeMode::ForwardMode:
      newFunc = Logic.CreateForwardDiff(
          fn, retType, constants, TA,
          /*should return*/ false, mode, freeMemory, width,
          /*addedType*/ nullptr, type_args, volatile_args,
          /*augmented*/ nullptr);
      break;
    case DerivativeMode::ForwardModeSplit: {
      bool forceAnonymousTape = !sizeOnly && allocatedTapeSize == -1;
      aug = &Logic.CreateAugmentedPrimal(
          fn, retType, constants, TA,
          /*returnUsed*/ false, /*shadowReturnUsed*/ false, type_args,
          volatile_args, forceAnonymousTape, width, /*atomicAdd*/ AtomicAdd);
      auto &DL = fn->getParent()->getDataLayout();
      if (!forceAnonymousTape) {
        assert(!aug->tapeType);
        if (aug->returns.find(AugmentedStruct::Tape) != aug->returns.end()) {
          auto tapeIdx = aug->returns.find(AugmentedStruct::Tape)->second;
          tapeType = (tapeIdx == -1)
                         ? aug->fn->getReturnType()
                         : cast<StructType>(aug->fn->getReturnType())
                               ->getElementType(tapeIdx);
        } else {
          if (sizeOnly) {
            CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 0, false));
            CI->eraseFromParent();
            return true;
          }
        }
        if (sizeOnly) {
          auto size = DL.getTypeSizeInBits(tapeType) / 8;
          CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), size, false));
          CI->eraseFromParent();
          return true;
        }
        if (tapeType &&
            DL.getTypeSizeInBits(tapeType) < 8 * (size_t)allocatedTapeSize) {
          auto bytes = DL.getTypeSizeInBits(tapeType) / 8;
          EmitFailure("Insufficient tape allocation size", CI->getDebugLoc(),
                      CI, "need ", bytes, " bytes have ", allocatedTapeSize,
                      " bytes");
        }
      } else {
        tapeType = PointerType::getInt8PtrTy(fn->getContext());
      }
      newFunc = Logic.CreateForwardDiff(
          fn, retType, constants, TA,
          /*should return*/ false, mode, freeMemory, width,
          /*addedType*/ tapeType, type_args, volatile_args, aug);
      break;
    }
    case DerivativeMode::ReverseModeCombined:
      assert(freeMemory);
      newFunc = Logic.CreatePrimalAndGradient(
          (ReverseCacheKey){.todiff = fn,
                            .retType = retType,
                            .constant_args = constants,
                            .uncacheable_args = volatile_args,
                            .returnUsed = false,
                            .shadowReturnUsed = false,
                            .mode = mode,
                            .width = width,
                            .freeMemory = freeMemory,
                            .AtomicAdd = AtomicAdd,
                            .additionalType = nullptr,
                            .typeInfo = type_args},
          TA, /*augmented*/ nullptr);
      break;
    case DerivativeMode::ReverseModePrimal:
    case DerivativeMode::ReverseModeGradient: {
      bool forceAnonymousTape = !sizeOnly && allocatedTapeSize == -1;
      bool shadowReturnUsed = returnUsed && (retType == DIFFE_TYPE::DUP_ARG ||
                                             retType == DIFFE_TYPE::DUP_NONEED);
      aug = &Logic.CreateAugmentedPrimal(
          fn, retType, constants, TA, returnUsed, shadowReturnUsed, type_args,
          volatile_args, forceAnonymousTape, width,
          /*atomicAdd*/ AtomicAdd);
      auto &DL = fn->getParent()->getDataLayout();
      if (!forceAnonymousTape) {
        assert(!aug->tapeType);
        if (aug->returns.find(AugmentedStruct::Tape) != aug->returns.end()) {
          auto tapeIdx = aug->returns.find(AugmentedStruct::Tape)->second;
          tapeType = (tapeIdx == -1)
                         ? aug->fn->getReturnType()
                         : cast<StructType>(aug->fn->getReturnType())
                               ->getElementType(tapeIdx);
        } else {
          if (sizeOnly) {
            CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 0, false));
            CI->eraseFromParent();
            return true;
          }
        }
        if (sizeOnly) {
          auto size = DL.getTypeSizeInBits(tapeType) / 8;
          CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), size, false));
          CI->eraseFromParent();
          return true;
        }
        if (tapeType &&
            DL.getTypeSizeInBits(tapeType) < 8 * (size_t)allocatedTapeSize) {
          auto bytes = DL.getTypeSizeInBits(tapeType) / 8;
          EmitFailure("Insufficient tape allocation size", CI->getDebugLoc(),
                      CI, "need ", bytes, " bytes have ", allocatedTapeSize,
                      " bytes");
        }
      } else {
        tapeType = PointerType::getInt8PtrTy(fn->getContext());
      }
      if (mode == DerivativeMode::ReverseModePrimal)
        newFunc = aug->fn;
      else
        newFunc = Logic.CreatePrimalAndGradient(
            (ReverseCacheKey){.todiff = fn,
                              .retType = retType,
                              .constant_args = constants,
                              .uncacheable_args = volatile_args,
                              .returnUsed = false,
                              .shadowReturnUsed = false,
                              .mode = mode,
                              .width = width,
                              .freeMemory = freeMemory,
                              .AtomicAdd = AtomicAdd,
                              .additionalType = tapeType,
                              .typeInfo = type_args},
            TA, aug);
    }
    }

    if (!newFunc)
      return false;

    if (differentialReturn) {
      if (differet)
        args.push_back(differet);
      else if (fn->getReturnType()->isFPOrFPVectorTy()) {
        Constant *seed = ConstantFP::get(fn->getReturnType(), 1.0);
        if (width == 1) {
          args.push_back(seed);
        } else {
          ArrayType *arrayType = ArrayType::get(fn->getReturnType(), width);
          args.push_back(ConstantArray::get(
              arrayType, SmallVector<Constant *, 3>(width, seed)));
        }
      } else if (auto ST = dyn_cast<StructType>(fn->getReturnType())) {
        SmallVector<Constant *, 2> csts;
        for (auto e : ST->elements()) {
          csts.push_back(ConstantFP::get(e, 1.0));
        }
        args.push_back(ConstantStruct::get(ST, csts));
      }
    }

    if ((mode == DerivativeMode::ReverseModeGradient ||
         mode == DerivativeMode::ForwardModeSplit) &&
        tape && tapeType) {
      auto &DL = fn->getParent()->getDataLayout();
      if (tapeIsPointer) {
        tape = Builder.CreateBitCast(
            tape, PointerType::get(
                      tapeType,
                      cast<PointerType>(tape->getType())->getAddressSpace()));
#if LLVM_VERSION_MAJOR > 7
        tape = Builder.CreateLoad(tapeType, tape);
#else
        tape = Builder.CreateLoad(tape);
#endif
      } else if (tapeType != tape->getType() &&
                 DL.getTypeSizeInBits(tapeType) <=
                     DL.getTypeSizeInBits(tape->getType())) {
        IRBuilder<> EB(&CI->getParent()->getParent()->getEntryBlock().front());
        auto AL = EB.CreateAlloca(tape->getType());
        Builder.CreateStore(tape, AL);
#if LLVM_VERSION_MAJOR > 7
        tape = Builder.CreateLoad(
            tapeType,
            Builder.CreatePointerCast(AL, PointerType::getUnqual(tapeType)));
#else
        tape = Builder.CreateLoad(
            Builder.CreatePointerCast(AL, PointerType::getUnqual(tapeType)));
#endif
      }
      assert(tape->getType() == tapeType);
      args.push_back(tape);
    }
    assert(newFunc);

    if (EnzymePrint) {
      llvm::errs() << "postfn:\n" << *newFunc << "\n";
    }
    Builder.setFastMathFlags(getFast());

    // call newFunc with the provided arguments.
    if (args.size() != newFunc->getFunctionType()->getNumParams()) {
      llvm::errs() << *CI << "\n";
      llvm::errs() << *newFunc << "\n";
      for (auto arg : args) {
        llvm::errs() << " + " << *arg << "\n";
      }
      auto modestr = to_string(mode);
      EmitFailure(
          "TooFewArguments", CI->getDebugLoc(), CI,
          "Too few arguments passed to __enzyme_autodiff mode=", modestr);
      return false;
    }
    assert(args.size() == newFunc->getFunctionType()->getNumParams());
    CallInst *diffretc = cast<CallInst>(Builder.CreateCall(newFunc, args));
    diffretc->setCallingConv(CI->getCallingConv());
    diffretc->setDebugLoc(CI->getDebugLoc());
#if LLVM_VERSION_MAJOR >= 9
    for (auto pair : byVal) {
      diffretc->addParamAttr(
          pair.first,
          Attribute::getWithByValType(diffretc->getContext(), pair.second));
    }
#endif
    Value *diffret = diffretc;
    if (mode == DerivativeMode::ReverseModePrimal && tape) {
      if (aug->returns.find(AugmentedStruct::Tape) != aug->returns.end()) {
        auto tapeIdx = aug->returns.find(AugmentedStruct::Tape)->second;
        tapeType = (tapeIdx == -1) ? aug->fn->getReturnType()
                                   : cast<StructType>(aug->fn->getReturnType())
                                         ->getElementType(tapeIdx);
        unsigned idxs[] = {(unsigned)tapeIdx};
        Value *tapeRes = (tapeIdx == -1)
                             ? diffret
                             : Builder.CreateExtractValue(diffret, idxs);
        Builder.CreateStore(
            tapeRes,
            Builder.CreateBitCast(
                tape,
                PointerType::get(
                    tapeRes->getType(),
                    cast<PointerType>(tape->getType())->getAddressSpace())));
        if (tapeIdx != -1) {
          auto ST = cast<StructType>(diffret->getType());
          SmallVector<Type *, 2> tys(ST->elements().begin(),
                                     ST->elements().end());
          tys.erase(tys.begin());
          auto ST0 = StructType::get(ST->getContext(), tys);
          Value *out = UndefValue::get(ST0);
          for (unsigned i = 0; i < tys.size(); i++) {
            unsigned in_idx[] = {i};
            unsigned out_idx[] = {i + 1};
            out = Builder.CreateInsertValue(
                out, Builder.CreateExtractValue(diffret, out_idx), in_idx);
          }
          diffret = out;
        } else {
          auto ST0 = StructType::get(tape->getContext(), {});
          diffret = UndefValue::get(ST0);
        }
      }
    }
    StructType *CIsty = dyn_cast<StructType>(CI->getType());
    StructType *diffretsty = dyn_cast<StructType>(diffret->getType());

    // Adapt the returned vector type to the struct type expected by our calling
    // convention.
    if (width > 1 && !diffret->getType()->isEmptyTy() &&
        !diffret->getType()->isVoidTy() &&
        (mode == DerivativeMode::ForwardMode ||
         mode == DerivativeMode::ForwardModeSplit)) {

      /// Actual return type (including struct return)
      Type *returnType =
          CI->hasStructRetAttr()
              ? dyn_cast<PointerType>(CI->getArgOperand(0)->getType())
                    ->getPointerElementType()
              : CI->getType();

      if (StructType *sty = dyn_cast<StructType>(returnType)) {
        Value *agg = ConstantAggregateZero::get(sty);

        for (unsigned int i = 0; i < width; i++) {
          Value *elem = Builder.CreateExtractValue(diffret, {i});
#if LLVM_VERSION_MAJOR >= 11
          if (auto vty = dyn_cast<FixedVectorType>(elem->getType())) {
#else
          if (auto vty = dyn_cast<VectorType>(elem->getType())) {
#endif
            for (unsigned j = 0; j < vty->getNumElements(); ++j) {
              Value *vecelem = Builder.CreateExtractElement(elem, j);
              agg = Builder.CreateInsertValue(agg, vecelem, {i * j});
            }
          } else {
            agg = Builder.CreateInsertValue(agg, elem, {i});
          }
        }
        diffret = agg;
      }
    }

    if (!diffret->getType()->isEmptyTy() && !diffret->getType()->isVoidTy() &&
        !CI->getType()->isEmptyTy() &&
        (!CI->getType()->isVoidTy() || CI->hasStructRetAttr())) {
      if (diffret->getType() == CI->getType()) {
        CI->replaceAllUsesWith(diffret);
      } else if (CIsty && diffretsty && CIsty->isLayoutIdentical(diffretsty)) {
        IRBuilder<> Builder(CI);
        Value *newStruct = UndefValue::get(CIsty);
        for (unsigned int i = 0; i < CIsty->getStructNumElements(); i++) {
          Value *elem = Builder.CreateExtractValue(diffret, {i});
          newStruct = Builder.CreateInsertValue(newStruct, elem, {i});
        }
        CI->replaceAllUsesWith(newStruct);
      } else if (mode == DerivativeMode::ReverseModePrimal) {
        auto &DL = fn->getParent()->getDataLayout();
        if (DL.getTypeSizeInBits(CI->getType()) >=
            DL.getTypeSizeInBits(diffret->getType())) {
          IRBuilder<> EB(
              &CI->getParent()->getParent()->getEntryBlock().front());
          auto AL = EB.CreateAlloca(CI->getType());
          Builder.CreateStore(
              diffret, Builder.CreatePointerCast(
                           AL, PointerType::getUnqual(diffret->getType())));
#if LLVM_VERSION_MAJOR > 7
          Value *cload = Builder.CreateLoad(CI->getType(), AL);
#else
          Value *cload = Builder.CreateLoad(AL);
#endif
          CI->replaceAllUsesWith(cload);
        } else {
          llvm::errs() << *CI << " - " << *diffret << "\n";
          assert(0 && " what");
        }
      } else if (CI->hasStructRetAttr()) {
        Value *sret = CI->getArgOperand(0);

        // Assign results to struct allocated at the call site.
        if (StructType *st = cast<StructType>(diffret->getType())) {
          for (unsigned int i = 0; i < st->getNumElements(); i++) {
#if LLVM_VERSION_MAJOR > 7
            Value *sgep = Builder.CreateStructGEP(
                sret->getType()->getPointerElementType(), sret, i);
#else
            Value *sgep = Builder.CreateStructGEP(sret, i);
#endif
            Builder.CreateStore(Builder.CreateExtractValue(diffret, {i}), sgep);
          }
        }
      } else {

        unsigned idxs[] = {0};
        auto diffreti = Builder.CreateExtractValue(diffret, idxs);
        if (diffreti->getType() == CI->getType()) {
          CI->replaceAllUsesWith(diffreti);
        } else if (diffret->getType() == CI->getType()) {
          CI->replaceAllUsesWith(diffret);
        } else {
          EmitFailure("IllegalReturnCast", CI->getDebugLoc(), CI,
                      "Cannot cast return type of gradient ",
                      *diffreti->getType(), *diffreti, ", to desired type ",
                      *CI->getType());
          return false;
        }
      }
    } else {
      CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
    }
    CI->eraseFromParent();

    if (Logic.PostOpt) {
#if LLVM_VERSION_MAJOR >= 11
      auto Params = llvm::getInlineParams();

      llvm::SetVector<CallInst *> Q;
      Q.insert(diffretc);
      TargetLibraryInfoWrapperPass TLIWP(
          Triple(newFunc->getParent()->getTargetTriple()));
      while (Q.size()) {
        auto cur = *Q.begin();
        Function *outerFunc = cur->getParent()->getParent();
        llvm::OptimizationRemarkEmitter ORE(outerFunc);
        Q.erase(Q.begin());
        if (auto F = cur->getCalledFunction()) {
          if (!F->empty()) {
            // Garbage collect AC's created
            SmallVector<AssumptionCache *, 2> ACAlloc;
            auto getAC = [&](Function &F) -> llvm::AssumptionCache & {
              auto AC = new AssumptionCache(F);
              ACAlloc.push_back(AC);
              return *AC;
            };
            auto GetTLI =
                [&](llvm::Function &F) -> const llvm::TargetLibraryInfo & {
              return TLIWP.getTLI(F);
            };

            auto GetInlineCost = [&](CallBase &CB) {
              TargetTransformInfo TTI(F->getParent()->getDataLayout());
              auto cst = llvm::getInlineCost(CB, Params, TTI, getAC, GetTLI);
              return cst;
            };
            if (llvm::shouldInline(*cur, GetInlineCost, ORE)) {
              InlineFunctionInfo IFI;
              InlineResult IR =
#if LLVM_VERSION_MAJOR >= 11
                  InlineFunction(*cur, IFI);
#else
                  InlineFunction(cur, IFI);
#endif
              if (IR.isSuccess()) {
                for (auto U : outerFunc->users()) {
                  if (auto CI = dyn_cast<CallInst>(U)) {
                    if (CI->getCalledFunction() == outerFunc) {
                      Q.insert(CI);
                    }
                  }
                }
              }
            }
            for (auto AC : ACAlloc) {
              delete AC;
            }
          }
        }
      }
#endif
    }
    return true;
  }

  bool lowerEnzymeCalls(Function &F, bool &successful,
                        std::set<Function *> &done) {
    if (done.count(&F))
      return false;
    done.insert(&F);

    if (F.empty())
      return false;

#if LLVM_VERSION_MAJOR >= 10
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
#else
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
#endif

    bool Changed = false;

    for (BasicBlock &BB : F)
      if (InvokeInst *II = dyn_cast<InvokeInst>(BB.getTerminator())) {

        Function *Fn = II->getCalledFunction();

#if LLVM_VERSION_MAJOR >= 11
        if (auto castinst = dyn_cast<ConstantExpr>(II->getCalledOperand()))
#else
        if (auto castinst = dyn_cast<ConstantExpr>(II->getCalledValue()))
#endif
        {
          if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
              Fn = fn;
        }
        if (!Fn)
          continue;

        if (!(Fn->getName().contains("__enzyme_float") ||
              Fn->getName().contains("__enzyme_double") ||
              Fn->getName().contains("__enzyme_integer") ||
              Fn->getName().contains("__enzyme_pointer") ||
              Fn->getName().contains("__enzyme_virtualreverse") ||
              Fn->getName().contains("__enzyme_call_inactive") ||
              Fn->getName().contains("__enzyme_autodiff") ||
              Fn->getName().contains("__enzyme_fwddiff") ||
              Fn->getName().contains("__enzyme_fwdsplit") ||
              Fn->getName().contains("__enzyme_augmentfwd") ||
              Fn->getName().contains("__enzyme_augmentsize") ||
              Fn->getName().contains("__enzyme_reverse")))
          continue;

        SmallVector<Value *, 16> CallArgs(II->arg_begin(), II->arg_end());
        SmallVector<OperandBundleDef, 1> OpBundles;
        II->getOperandBundlesAsDefs(OpBundles);
// Insert a normal call instruction...
#if LLVM_VERSION_MAJOR >= 8
        CallInst *NewCall =
            CallInst::Create(II->getFunctionType(), II->getCalledOperand(),
                             CallArgs, OpBundles, "", II);
#else
        CallInst *NewCall =
            CallInst::Create(II->getFunctionType(), II->getCalledValue(),
                             CallArgs, OpBundles, "", II);
#endif
        NewCall->takeName(II);
        NewCall->setCallingConv(II->getCallingConv());
        NewCall->setAttributes(II->getAttributes());
        NewCall->setDebugLoc(II->getDebugLoc());
        II->replaceAllUsesWith(NewCall);

        // Insert an unconditional branch to the normal destination.
        BranchInst::Create(II->getNormalDest(), II);

        // Remove any PHI node entries from the exception destination.
        II->getUnwindDest()->removePredecessor(&BB);

        // Remove the invoke instruction now.
        BB.getInstList().erase(II);
        Changed = true;
      }

    MapVector<CallInst *, DerivativeMode> toLower;
    MapVector<CallInst *, DerivativeMode> toVirtual;
    MapVector<CallInst *, DerivativeMode> toSize;
    SetVector<CallInst *> InactiveCalls;
    SetVector<CallInst *> IterCalls;
  retry:;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        CallInst *CI = dyn_cast<CallInst>(&I);

        if (!CI)
          continue;

        Function *Fn = CI->getCalledFunction();

#if LLVM_VERSION_MAJOR >= 11
        if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
        if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
        {
          if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
              Fn = fn;
        }

        if (!Fn)
          continue;

#if LLVM_VERSION_MAJOR >= 14
        size_t num_args = CI->arg_size();
#else
        size_t num_args = CI->getNumArgOperands();
#endif

        if (Fn->getName().contains("__enzyme_float")) {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_integer")) {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_double")) {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_pointer")) {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_virtualreverse")) {
          Fn->addFnAttr(Attribute::ReadNone);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
        }
        if (Fn->getName().contains("__enzyme_iter")) {
          Fn->addFnAttr(Attribute::ReadNone);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
        }
        if (Fn->getName().contains("__enzyme_call_inactive")) {
          InactiveCalls.insert(CI);
        }
        if (Fn->getName() == "omp_get_max_threads" ||
            Fn->getName() == "omp_get_thread_num") {
          Fn->addFnAttr(Attribute::ReadOnly);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
          Fn->addFnAttr(Attribute::InaccessibleMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOnly);
        }
        if ((Fn->getName() == "cblas_ddot" || Fn->getName() == "cblas_sdot") &&
            Fn->isDeclaration()) {
          Fn->addFnAttr(Attribute::ReadOnly);
          Fn->addFnAttr(Attribute::ArgMemOnly);
          CI->addParamAttr(1, Attribute::ReadOnly);
          CI->addParamAttr(1, Attribute::NoCapture);
          CI->addParamAttr(3, Attribute::ReadOnly);
          CI->addParamAttr(3, Attribute::NoCapture);
        }
        if (Fn->getName() == "frexp" || Fn->getName() == "frexpf" ||
            Fn->getName() == "frexpl") {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ArgMemOnly);
          CI->addParamAttr(1, Attribute::WriteOnly);
        }
        if (Fn->getName() == "__fd_sincos_1" || Fn->getName() == "__fd_cos_1" ||
            Fn->getName() == "__mth_i_ipowi") {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
        }
        if (Fn->getName() == "f90io_fmtw_end" ||
            Fn->getName() == "f90io_unf_end") {
          Fn->addFnAttr(Attribute::InaccessibleMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOnly);
        }
        if (Fn->getName() == "f90io_open2003a") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          for (size_t i : {0, 1, 2, 3, 4, 5, 6, 7, /*8, */ 9, 10, 11, 12, 13}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }
          // todo more
          for (size_t i : {0, 1}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName() == "f90io_fmtw_inita") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          // todo more
          for (size_t i : {0, 2}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }

          // todo more
          for (size_t i : {0, 2}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }

        if (Fn->getName() == "f90io_unf_init") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          // todo more
          for (size_t i : {0, 1, 2, 3}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }

          // todo more
          for (size_t i : {0, 1, 2, 3}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }

        if (Fn->getName() == "f90io_src_info03a") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          // todo more
          for (size_t i : {0, 1}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }

          // todo more
          for (size_t i : {0}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName() == "f90io_sc_d_fmt_write" ||
            Fn->getName() == "f90io_sc_i_fmt_write" ||
            Fn->getName() == "ftnio_fmt_write64" ||
            Fn->getName() == "f90io_fmt_write64_aa" ||
            Fn->getName() == "f90io_fmt_writea" ||
            Fn->getName() == "f90io_unf_writea" ||
            Fn->getName() == "f90_pausea") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }

        bool enableEnzyme = false;
        bool virtualCall = false;
        bool sizeOnly = false;
        DerivativeMode mode;
        if (Fn->getName().contains("__enzyme_autodiff")) {
          enableEnzyme = true;
          mode = DerivativeMode::ReverseModeCombined;
        } else if (Fn->getName().contains("__enzyme_fwddiff")) {
          enableEnzyme = true;
          mode = DerivativeMode::ForwardMode;
        } else if (Fn->getName().contains("__enzyme_fwdsplit")) {
          enableEnzyme = true;
          mode = DerivativeMode::ForwardModeSplit;
        } else if (Fn->getName().contains("__enzyme_augmentfwd")) {
          enableEnzyme = true;
          mode = DerivativeMode::ReverseModePrimal;
        } else if (Fn->getName().contains("__enzyme_augmentsize")) {
          enableEnzyme = true;
          sizeOnly = true;
          mode = DerivativeMode::ReverseModePrimal;
        } else if (Fn->getName().contains("__enzyme_reverse")) {
          enableEnzyme = true;
          mode = DerivativeMode::ReverseModeGradient;
        } else if (Fn->getName().contains("__enzyme_virtualreverse")) {
          enableEnzyme = true;
          virtualCall = true;
          mode = DerivativeMode::ReverseModeCombined;
        }

        if (enableEnzyme) {

          Value *fn = CI->getArgOperand(0);
          while (auto ci = dyn_cast<CastInst>(fn)) {
            fn = ci->getOperand(0);
          }
          while (auto ci = dyn_cast<BlockAddress>(fn)) {
            fn = ci->getFunction();
          }
          while (auto ci = dyn_cast<ConstantExpr>(fn)) {
            fn = ci->getOperand(0);
          }
          if (auto si = dyn_cast<SelectInst>(fn)) {
            BasicBlock *post = BB.splitBasicBlock(CI);
            BasicBlock *sel1 = BasicBlock::Create(BB.getContext(), "sel1", &F);
            BasicBlock *sel2 = BasicBlock::Create(BB.getContext(), "sel2", &F);
            BB.getTerminator()->eraseFromParent();
            IRBuilder<> PB(&BB);
            PB.CreateCondBr(si->getCondition(), sel1, sel2);
            IRBuilder<> S1(sel1);
            auto B1 = S1.CreateBr(post);
            CallInst *cloned = cast<CallInst>(CI->clone());
            cloned->insertBefore(B1);
            cloned->setOperand(0, si->getTrueValue());
            IRBuilder<> S2(sel2);
            auto B2 = S2.CreateBr(post);
            CI->moveBefore(B2);
            CI->setOperand(0, si->getFalseValue());
            if (CI->getNumUses() != 0) {
              IRBuilder<> P(post->getFirstNonPHI());
              auto merge = P.CreatePHI(CI->getType(), 2);
              merge->addIncoming(cloned, sel1);
              merge->addIncoming(CI, sel2);
              CI->replaceAllUsesWith(merge);
            }
            goto retry;
          }

          if (virtualCall)
            toVirtual[CI] = mode;
          else if (sizeOnly)
            toSize[CI] = mode;
          else
            toLower[CI] = mode;

          if (auto dc = dyn_cast<Function>(fn)) {
            // Force postopt on any inner functions in the nested
            // AD case.
            bool tmp = Logic.PostOpt;
            Logic.PostOpt = true;
            Changed |= lowerEnzymeCalls(*dc, successful, done);
            Logic.PostOpt = tmp;
          }
        }
      }
    }

    for (auto CI : InactiveCalls) {
      IRBuilder<> B(CI);
      Value *fn = CI->getArgOperand(0);
      SmallVector<Value *, 4> Args;
      SmallVector<Type *, 4> ArgTypes;
#if LLVM_VERSION_MAJOR >= 14
      for (size_t i = 1; i < CI->arg_size(); ++i)
#else
      for (size_t i = 1; i < CI->getNumArgOperands(); ++i)
#endif
      {
        Args.push_back(CI->getArgOperand(i));
        ArgTypes.push_back(CI->getArgOperand(i)->getType());
      }
      auto FT = FunctionType::get(CI->getType(), ArgTypes, /*varargs*/ false);
      if (fn->getType() != FT) {
        fn = B.CreatePointerCast(fn, PointerType::getUnqual(FT));
      }
      auto Rep = B.CreateCall(FT, fn, Args);
      Rep->addAttribute(AttributeList::FunctionIndex,
                        Attribute::get(Rep->getContext(), "enzyme_inactive"));
      CI->replaceAllUsesWith(Rep);
      CI->eraseFromParent();
      Changed = true;
    }

    // Perform all the size replacements first to create constants
    for (auto pair : toSize) {
      successful &= HandleAutoDiff(pair.first, TLI, pair.second,
                                   /*sizeOnly*/ true);
      Changed = true;
      if (!successful)
        break;
    }
    for (auto pair : toLower) {
      successful &= HandleAutoDiff(pair.first, TLI, pair.second,
                                   /*sizeOnly*/ false);
      Changed = true;
      if (!successful)
        break;
    }

    for (auto pair : toVirtual) {
      auto CI = pair.first;
      Constant *fn = dyn_cast<Constant>(CI->getArgOperand(0));
      if (!fn) {
        EmitFailure("IllegalVirtual", CI->getDebugLoc(), CI,
                    "Cannot create virtual version of non-constant value ", *CI,
                    *CI->getArgOperand(0));
        return false;
      }
      TypeAnalysis TA(Logic.PPC.FAM);

      auto Arch =
          llvm::Triple(
              CI->getParent()->getParent()->getParent()->getTargetTriple())
              .getArch();

      bool AtomicAdd = Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
                       Arch == Triple::amdgcn;

      auto val = GradientUtils::GetOrCreateShadowConstant(
          Logic, TLI, TA, fn, pair.second, /*width*/ 1, AtomicAdd);
      CI->replaceAllUsesWith(ConstantExpr::getPointerCast(val, CI->getType()));
      CI->eraseFromParent();
      Changed = true;
    }

    if (Changed && EnzymeAttributor) {
      // TODO consider enabling when attributor does not delete
      // dead internal functions, which invalidates Enzyme's cache
      // code left here to re-enable upon Attributor patch

#if LLVM_VERSION_MAJOR >= 13 && !defined(FLANG) && !defined(ROCM)

      AnalysisGetter AG(Logic.PPC.FAM);
      SetVector<Function *> Functions;
      for (Function &F2 : *F.getParent()) {
        Functions.insert(&F2);
      }

      CallGraphUpdater CGUpdater;
      BumpPtrAllocator Allocator;
      InformationCache InfoCache(*F.getParent(), AG, Allocator,
                                 /* CGSCC */ nullptr);

      DenseSet<const char *> Allowed = {
          &AAHeapToStack::ID,     &AANoCapture::ID,

          &AAMemoryBehavior::ID,  &AAMemoryLocation::ID, &AANoUnwind::ID,
          &AANoSync::ID,          &AANoRecurse::ID,      &AAWillReturn::ID,
          &AANoReturn::ID,        &AANonNull::ID,        &AANoAlias::ID,
          &AADereferenceable::ID, &AAAlign::ID,

          &AAReturnedValues::ID,  &AANoFree::ID,         &AANoUndef::ID,

          //&AAValueSimplify::ID,
          //&AAReachability::ID,
          //&AAValueConstantRange::ID,
          //&AAUndefinedBehavior::ID,
          //&AAPotentialValues::ID,
      };

#if LLVM_VERSION_MAJOR >= 15
      AttributorConfig aconfig(CGUpdater);
      aconfig.Allowed = &Allowed;
      aconfig.DeleteFns = false;
      Attributor A(Functions, InfoCache, aconfig);
#else

      Attributor A(Functions, InfoCache, CGUpdater, &Allowed,
                   /*DeleteFns*/ false);
#endif
      for (Function *F : Functions) {
        // Populate the Attributor with abstract attribute opportunities in the
        // function and the information cache with IR information.
        A.identifyDefaultAbstractAttributes(*F);
      }
      A.run();
#endif
    }

    return Changed;
  }

  bool runOnModule(Module &M) override {
    constexpr static const char gradient_handler_name[] =
        "__enzyme_register_gradient";
    constexpr static const char derivative_handler_name[] =
        "__enzyme_register_derivative";
    constexpr static const char splitderivative_handler_name[] =
        "__enzyme_register_splitderivative";

    Logic.clear();

    bool changed = false;
    SmallVector<GlobalVariable *, 4> globalsToErase;
    for (GlobalVariable &g : M.globals()) {
      if (g.getName().contains(gradient_handler_name)) {
        handleCustomDerivative<gradient_handler_name,
                               DerivativeMode::ReverseModeGradient, 3>(
            M, g, globalsToErase);
      } else if (g.getName().contains(derivative_handler_name)) {
        handleCustomDerivative<derivative_handler_name,
                               DerivativeMode::ForwardMode, 2>(M, g,
                                                               globalsToErase);
      } else if (g.getName().contains(splitderivative_handler_name)) {
        handleCustomDerivative<splitderivative_handler_name,
                               DerivativeMode::ForwardModeSplit, 3>(
            M, g, globalsToErase);
      } else if (g.getName().contains("__enzyme_inactivefn")) {
        handleInactiveFunction(M, g, globalsToErase);
      }
    }
    for (auto g : globalsToErase) {
      g->eraseFromParent();
    }
    for (Function &F : M) {
      handleAnnotations(F);
      handleKnownFunctions(F);
      if (F.empty())
        continue;
      SmallVector<Instruction *, 4> toErase;
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto CI = dyn_cast<CallInst>(&I)) {
            Function *F = CI->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
            {
              if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                  F = fn;
                }
            }
            if (F && F->getName() == "f90_mzero8") {
              toErase.push_back(CI);
              IRBuilder<> B(CI);

              SmallVector<Value *, 4> args;
              args.push_back(CI->getArgOperand(0));
              args.push_back(
                  ConstantInt::get(Type::getInt8Ty(M.getContext()), 0));
              args.push_back(B.CreateMul(
                  CI->getArgOperand(1),
                  ConstantInt::get(CI->getArgOperand(1)->getType(), 8)));
#if LLVM_VERSION_MAJOR <= 6
              args.push_back(
                  ConstantInt::get(Type::getInt32Ty(M.getContext()), 1U));
#endif
              args.push_back(ConstantInt::getFalse(M.getContext()));

              Type *tys[] = {args[0]->getType(), args[2]->getType()};
              auto memsetIntr =
                  Intrinsic::getDeclaration(&M, Intrinsic::memset, tys);
              B.CreateCall(memsetIntr, args);
            }
          }
        }
      }
      for (Instruction *I : toErase) {
        I->eraseFromParent();
      }
    }

#if LLVM_VERSION_MAJOR >= 13
    if (Logic.PostOpt && EnzymeOMPOpt) {
      OpenMPOptPass().run(M, Logic.PPC.MAM);
      /// Attributor is run second time for promoted args to get attributes.
      AttributorPass().run(M, Logic.PPC.MAM);
      for (auto &F : M)
        if (!F.empty())
          PromotePass().run(F, Logic.PPC.FAM);
      changed = true;
    }
#endif

    std::set<Function *> done;
    for (Function &F : M) {
      if (F.empty())
        continue;

      bool successful = true;
      changed |= lowerEnzymeCalls(F, successful, done);

      if (!successful) {
        M.getContext().diagnose(
            (EnzymeFailure("FailedToDifferentiate", F.getSubprogram(),
                           &*F.getEntryBlock().begin())
             << "EnzymeFailure when replacing __enzyme_autodiff calls in "
             << F.getName()));
      }
    }

    SmallVector<CallInst *, 4> toErase;
    for (Function &F : M) {
      if (F.empty())
        continue;

      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto CI = dyn_cast<CallInst>(&I)) {
            Function *F = CI->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
            {
              if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                  F = fn;
                }
            }
            if (F) {
              if (F->getName().contains("__enzyme_float") ||
                  F->getName().contains("__enzyme_double") ||
                  F->getName().contains("__enzyme_integer") ||
                  F->getName().contains("__enzyme_pointer")) {
                toErase.push_back(CI);
              }
              if (F->getName() == "__enzyme_iter") {
                CI->replaceAllUsesWith(CI->getArgOperand(0));
                toErase.push_back(CI);
              }
            }
          }
        }
      }
    }
    for (auto I : toErase) {
      I->eraseFromParent();
      changed = true;
    }

    for (const auto &pair : Logic.PPC.cache)
      pair.second->eraseFromParent();
    Logic.clear();

    if (changed && Logic.PostOpt) {
      PassBuilder PB;
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;
      PB.registerModuleAnalyses(MAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
#if LLVM_VERSION_MAJOR >= 14
      auto PM = PB.buildModuleSimplificationPipeline(OptimizationLevel::O2,
                                                     ThinOrFullLTOPhase::None);
#elif LLVM_VERSION_MAJOR >= 12
      auto PM = PB.buildModuleSimplificationPipeline(
          PassBuilder::OptimizationLevel::O2, ThinOrFullLTOPhase::None);
#else
    auto PM = PB.buildModuleSimplificationPipeline(
        PassBuilder::OptimizationLevel::O2, PassBuilder::ThinLTOPhase::None);
#endif
      PM.run(M, MAM);
#if LLVM_VERSION_MAJOR >= 13
      if (EnzymeOMPOpt) {
        OpenMPOptPass().run(M, MAM);
        /// Attributor is run second time for promoted args to get attributes.
        AttributorPass().run(M, MAM);
        for (auto &F : M)
          if (!F.empty())
            PromotePass().run(F, FAM);
      }
#endif
    }
    return changed;
  }
};

} // namespace

char Enzyme::ID = 0;

static RegisterPass<Enzyme> X("enzyme", "Enzyme Pass");

ModulePass *createEnzymePass(bool PostOpt) { return new Enzyme(PostOpt); }

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddEnzymePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createEnzymePass(/*PostOpt*/ false));
}
