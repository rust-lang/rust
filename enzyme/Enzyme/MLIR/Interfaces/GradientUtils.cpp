//===- GradientUtils.cpp - Utilities for gradient interfaces --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/GradientUtils.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::enzyme;

static inline Type getShadowType(Type type, unsigned width = 1) {
  return type.cast<AutoDiffTypeInterface>().getShadowType(width);
}

mlir::enzyme::MGradientUtils::MGradientUtils(
    MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
    FunctionOpInterface oldFunc_, MTypeAnalysis &TA_, MTypeResults TR_,
    BlockAndValueMapping &invertedPointers_,
    const SmallPtrSetImpl<mlir::Value> &constantvalues_,
    const SmallPtrSetImpl<mlir::Value> &activevals_, DIFFE_TYPE ReturnActivity,
    ArrayRef<DIFFE_TYPE> ArgDiffeTypes_, BlockAndValueMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeMode mode, unsigned width, bool omp)
    : newFunc(newFunc_), Logic(Logic), mode(mode), oldFunc(oldFunc_), TA(TA_),
      TR(TR_), omp(omp), width(width), ArgDiffeTypes(ArgDiffeTypes_),
      originalToNewFn(originalToNewFn_),
      originalToNewFnOps(originalToNewFnOps_),
      invertedPointers(invertedPointers_) {

  /*
  for (BasicBlock &BB : *oldFunc) {
    for (Instruction &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        originalCalls.push_back(CI);
      }
    }
  }
  */

  /*
  for (BasicBlock &oBB : *oldFunc) {
    for (Instruction &oI : oBB) {
      newToOriginalFn[originalToNewFn[&oI]] = &oI;
    }
    newToOriginalFn[originalToNewFn[&oBB]] = &oBB;
  }
  for (Argument &oArg : oldFunc->args()) {
    newToOriginalFn[originalToNewFn[&oArg]] = &oArg;
  }
  */
  /*
  for (BasicBlock &BB : *newFunc) {
    originalBlocks.emplace_back(&BB);
  }
  tape = nullptr;
  tapeidx = 0;
  assert(originalBlocks.size() > 0);

  SmallVector<BasicBlock *, 4> ReturningBlocks;
  for (BasicBlock &BB : *oldFunc) {
    if (isa<ReturnInst>(BB.getTerminator()))
      ReturningBlocks.push_back(&BB);
  }
  for (BasicBlock &BB : *oldFunc) {
    bool legal = true;
    for (auto BRet : ReturningBlocks) {
      if (!(BRet == &BB || OrigDT.dominates(&BB, BRet))) {
        legal = false;
        break;
      }
    }
    if (legal)
      BlocksDominatingAllReturns.insert(&BB);
  }
  */
}

Value mlir::enzyme::MGradientUtils::getNewFromOriginal(
    const mlir::Value originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new val from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Block *
mlir::enzyme::MGradientUtils::getNewFromOriginal(mlir::Block *originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new blk from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Operation *
mlir::enzyme::MGradientUtils::getNewFromOriginal(Operation *originst) const {
  auto found = originalToNewFnOps.find(originst);
  if (found == originalToNewFnOps.end()) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    for (auto &pair : originalToNewFnOps) {
      llvm::errs() << " map[" << pair.first << "] = " << pair.second << "\n";
      // llvm::errs() << " map[" << pair.first << "] = " << pair.second << "
      // -- " << *pair.first << " " << *pair.second << "\n";
    }
    llvm::errs() << originst << " - " << *originst << "\n";
    llvm_unreachable("Could not get new op from original");
  }
  return found->second;
}

bool mlir::enzyme::MGradientUtils::isConstantValue(Value v) const {
  if (isa<mlir::IntegerType>(v.getType()))
    return true;
  if (isa<mlir::IndexType>(v.getType()))
    return true;

  if (matchPattern(v, m_Constant()))
    return true;

  // TODO
  return false;
}

Value mlir::enzyme::MGradientUtils::invertPointerM(Value v,
                                                   OpBuilder &Builder2) {
  // TODO
  if (invertedPointers.contains(v))
    return invertedPointers.lookupOrNull(v);

  if (isConstantValue(v)) {
    if (auto iface = v.getType().cast<AutoDiffTypeInterface>()) {
      OpBuilder::InsertionGuard guard(Builder2);
      Builder2.setInsertionPoint(getNewFromOriginal(v.getDefiningOp()));
      Value dv = iface.createNullValue(Builder2, v.getLoc());
      invertedPointers.map(v, dv);
      return dv;
    }
    return getNewFromOriginal(v);
  }
  llvm::errs() << " could not invert pointer v " << v << "\n";
  llvm_unreachable("could not invert pointer");
}

void mlir::enzyme::MGradientUtils::setDiffe(mlir::Value val, mlir::Value toset,
                                            OpBuilder &BuilderM) {
  /*
 if (auto arg = dyn_cast<Argument>(val))
   assert(arg->getParent() == oldFunc);
 if (auto inst = dyn_cast<Instruction>(val))
   assert(inst->getParent()->getParent() == oldFunc);
   */
  if (isConstantValue(val)) {
    llvm::errs() << newFunc << "\n";
    llvm::errs() << val << "\n";
  }
  assert(!isConstantValue(val));
  if (mode == DerivativeMode::ForwardMode ||
      mode == DerivativeMode::ForwardModeSplit) {
    assert(getShadowType(val.getType()) == toset.getType());
    auto found = invertedPointers.lookupOrNull(val);
    assert(found != nullptr);
    auto placeholder = found.getDefiningOp<enzyme::PlaceholderOp>();
    invertedPointers.erase(val);
    // replaceAWithB(placeholder, toset);
    placeholder.replaceAllUsesWith(toset);
    erase(placeholder);
    invertedPointers.map(val, toset);
    return;
  }
  /*
  Value *tostore = getDifferential(val);
  if (toset->getType() != tostore->getType()->getPointerElementType()) {
    llvm::errs() << "toset:" << *toset << "\n";
    llvm::errs() << "tostore:" << *tostore << "\n";
  }
  assert(toset->getType() == tostore->getType()->getPointerElementType());
  BuilderM.CreateStore(toset, tostore);
  */
}

void mlir::enzyme::MGradientUtils::forceAugmentedReturns() {
  // TODO also block arguments
  // assert(TR.getFunction() == oldFunc);

  // Don't create derivatives for code that results in termination
  // if (notForAnalysis.find(&oBB) != notForAnalysis.end())
  //  continue;

  // LoopContext loopContext;
  // getContext(cast<BasicBlock>(getNewFromOriginal(&oBB)), loopContext);

  oldFunc.walk([&](Block *blk) {
    if (blk == &oldFunc.getBody().getBlocks().front())
      return;
    auto nblk = getNewFromOriginal(blk);
    for (auto val : llvm::reverse(blk->getArguments())) {
      if (isConstantValue(val))
        continue;
      auto i = val.getArgNumber();
      mlir::Value dval;
      if (i == blk->getArguments().size() - 1)
        dval = nblk->addArgument(getShadowType(val.getType()), val.getLoc());
      else
        dval = nblk->insertArgument(nblk->args_begin() + i + 1,
                                    getShadowType(val.getType()), val.getLoc());

      invertedPointers.map(val, dval);
    }
  });

  oldFunc.walk([&](Operation *inst) {
    if (inst == oldFunc)
      return;
    if (mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeSplit) {
      OpBuilder BuilderZ(getNewFromOriginal(inst));
      for (auto res : inst->getResults()) {
        if (!isConstantValue(res)) {
          mlir::Type antiTy = getShadowType(res.getType());
          auto anti = BuilderZ.create<enzyme::PlaceholderOp>(res.getLoc(),
                                                             res.getType());
          invertedPointers.map(res, anti);
        }
      }
      return;
    }
    /*

    if (inst->getType()->isFPOrFPVectorTy())
      continue; //! op->getType()->isPointerTy() &&
                //! !op->getType()->isIntegerTy()) {

    if (!TR.query(inst)[{-1}].isPossiblePointer())
      continue;

    if (isa<LoadInst>(inst)) {
      IRBuilder<> BuilderZ(inst);
      getForwardBuilder(BuilderZ);
      Type *antiTy = getShadowType(inst->getType());
      PHINode *anti =
          BuilderZ.CreatePHI(antiTy, 1, inst->getName() + "'il_phi");
      invertedPointers.insert(std::make_pair(
          (const Value *)inst, InvertedPointerVH(this, anti)));
      continue;
    }

    if (!isa<CallInst>(inst)) {
      continue;
    }

    if (isa<IntrinsicInst>(inst)) {
      continue;
    }

    if (isConstantValue(inst)) {
      continue;
    }

    CallInst *op = cast<CallInst>(inst);
    Function *called = op->getCalledFunction();

    IRBuilder<> BuilderZ(inst);
    getForwardBuilder(BuilderZ);
    Type *antiTy = getShadowType(inst->getType());

    PHINode *anti =
        BuilderZ.CreatePHI(antiTy, 1, op->getName() + "'ip_phi");
    invertedPointers.insert(
        std::make_pair((const Value *)inst, InvertedPointerVH(this, anti)));

    if (called && isAllocationFunction(called->getName(), TLI)) {
      anti->setName(op->getName() + "'mi");
    }
    */
  });
}

LogicalResult MGradientUtils::visitChild(Operation *op) {
  if (mode == DerivativeMode::ForwardMode) {
    if (auto iface = dyn_cast<AutoDiffOpInterface>(op)) {
      OpBuilder builder(op->getContext());
      builder.setInsertionPoint(getNewFromOriginal(op));
      return iface.createForwardModeAdjoint(builder, this);
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

namespace {

mlir::FunctionType getFunctionTypeForClone(
    mlir::FunctionType FTy, DerivativeMode mode, unsigned width,
    mlir::Type additionalArg, llvm::ArrayRef<DIFFE_TYPE> constant_args,
    bool diffeReturnArg, ReturnType returnValue, DIFFE_TYPE returnType) {
  SmallVector<mlir::Type, 4> RetTypes;
  if (returnValue == ReturnType::ArgsWithReturn ||
      returnValue == ReturnType::Return) {
    assert(FTy.getNumResults() == 1);
    if (returnType != DIFFE_TYPE::CONSTANT &&
        returnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(FTy.getResult(0), width));
    } else {
      RetTypes.push_back(FTy.getResult(0));
    }
  } else if (returnValue == ReturnType::ArgsWithTwoReturns ||
             returnValue == ReturnType::TwoReturns) {
    assert(FTy.getNumResults() == 1);
    RetTypes.push_back(FTy.getResult(0));
    if (returnType != DIFFE_TYPE::CONSTANT &&
        returnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(FTy.getResult(0), width));
    } else {
      RetTypes.push_back(FTy.getResult(0));
    }
  }

  SmallVector<mlir::Type, 4> ArgTypes;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  unsigned argno = 0;

  for (auto I : FTy.getInputs()) {
    ArgTypes.push_back(I);
    if (constant_args[argno] == DIFFE_TYPE::DUP_ARG ||
        constant_args[argno] == DIFFE_TYPE::DUP_NONEED) {
      ArgTypes.push_back(getShadowType(I, width));
    } else if (constant_args[argno] == DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(I, width));
    }
    ++argno;
  }

  if (diffeReturnArg) {
    ArgTypes.push_back(getShadowType(FTy.getResult(0), width));
  }
  if (additionalArg) {
    ArgTypes.push_back(additionalArg);
  }

  OpBuilder builder(FTy.getContext());
  if (returnValue == ReturnType::TapeAndTwoReturns ||
      returnValue == ReturnType::TapeAndReturn ||
      returnValue == ReturnType::Tape) {
    RetTypes.insert(RetTypes.begin(),
                    LLVM::LLVMPointerType::get(builder.getIntegerType(8)));
  }

  // Create a new function type...
  return builder.getFunctionType(ArgTypes, RetTypes);
}

void cloneInto(Region *src, Region *dest, Region::iterator destPos,
               BlockAndValueMapping &mapper,
               std::map<Operation *, Operation *> &opMap);
void cloneInto(Region *src, Region *dest, BlockAndValueMapping &mapper,
               std::map<Operation *, Operation *> &opMap) {
  cloneInto(src, dest, dest->end(), mapper, opMap);
}
Operation *clone(Operation *src, BlockAndValueMapping &mapper,
                 Operation::CloneOptions options,
                 std::map<Operation *, Operation *> &opMap) {
  SmallVector<Value, 8> operands;
  SmallVector<Block *, 2> successors;

  // Remap the operands.
  if (options.shouldCloneOperands()) {
    operands.reserve(src->getNumOperands());
    for (auto opValue : src->getOperands())
      operands.push_back(mapper.lookupOrDefault(opValue));
  }

  // Remap the successors.
  successors.reserve(src->getNumSuccessors());
  for (Block *successor : src->getSuccessors())
    successors.push_back(mapper.lookupOrDefault(successor));

  // Create the new operation.
  auto *newOp =
      src->create(src->getLoc(), src->getName(), src->getResultTypes(),
                  operands, src->getAttrs(), successors, src->getNumRegions());

  // Clone the regions.
  if (options.shouldCloneRegions()) {
    for (unsigned i = 0; i != src->getNumRegions(); ++i)
      cloneInto(&src->getRegion(i), &newOp->getRegion(i), mapper, opMap);
  }

  // Remember the mapping of any results.
  for (unsigned i = 0, e = src->getNumResults(); i != e; ++i)
    mapper.map(src->getResult(i), newOp->getResult(i));

  opMap[src] = newOp;
  return newOp;
}
/// Clone this region into 'dest' before the given position in 'dest'.
void cloneInto(Region *src, Region *dest, Region::iterator destPos,
               BlockAndValueMapping &mapper,
               std::map<Operation *, Operation *> &opMap) {
  assert(src);
  assert(dest && "expected valid region to clone into");
  assert(src != dest && "cannot clone region into itself");

  // If the list is empty there is nothing to clone.
  if (src->empty())
    return;

  // The below clone implementation takes special care to be read only for the
  // sake of multi threading. That essentially means not adding any uses to any
  // of the blocks or operation results contained within this region as that
  // would lead to a write in their use-def list. This is unavoidable for
  // 'Value's from outside the region however, in which case it is not read
  // only. Using the BlockAndValueMapper it is possible to remap such 'Value's
  // to ones owned by the calling thread however, making it read only once
  // again.

  // First clone all the blocks and block arguments and map them, but don't yet
  // clone the operations, as they may otherwise add a use to a block that has
  // not yet been mapped
  for (Block &block : *src) {
    Block *newBlock = new Block();
    mapper.map(&block, newBlock);

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    for (auto arg : block.getArguments())
      if (!mapper.contains(arg))
        mapper.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));

    dest->getBlocks().insert(destPos, newBlock);
  }

  auto newBlocksRange =
      llvm::make_range(Region::iterator(mapper.lookup(&src->front())), destPos);

  // Now follow up with creating the operations, but don't yet clone their
  // regions, nor set their operands. Setting the successors is safe as all have
  // already been mapped. We are essentially just creating the operation results
  // to be able to map them.
  // Cloning the operands and region as well would lead to uses of operations
  // not yet mapped.
  auto cloneOptions =
      Operation::CloneOptions::all().cloneRegions(false).cloneOperands(false);
  for (auto zippedBlocks : llvm::zip(*src, newBlocksRange)) {
    Block &sourceBlock = std::get<0>(zippedBlocks);
    Block &clonedBlock = std::get<1>(zippedBlocks);
    // Clone and remap the operations within this block.
    for (Operation &op : sourceBlock) {
      clonedBlock.push_back(clone(&op, mapper, cloneOptions, opMap));
    }
  }

  // Finally now that all operation results have been mapped, set the operands
  // and clone the regions.
  SmallVector<Value> operands;
  for (auto zippedBlocks : llvm::zip(*src, newBlocksRange)) {
    for (auto ops :
         llvm::zip(std::get<0>(zippedBlocks), std::get<1>(zippedBlocks))) {
      Operation &source = std::get<0>(ops);
      Operation &clone = std::get<1>(ops);

      operands.resize(source.getNumOperands());
      llvm::transform(
          source.getOperands(), operands.begin(),
          [&](Value operand) { return mapper.lookupOrDefault(operand); });
      clone.setOperands(operands);

      for (auto regions : llvm::zip(source.getRegions(), clone.getRegions()))
        cloneInto(&std::get<0>(regions), &std::get<1>(regions), mapper, opMap);
    }
  }
}

FunctionOpInterface CloneFunctionWithReturns(
    DerivativeMode mode, unsigned width, FunctionOpInterface F,
    BlockAndValueMapping &ptrInputs, ArrayRef<DIFFE_TYPE> constant_args,
    SmallPtrSetImpl<mlir::Value> &constants,
    SmallPtrSetImpl<mlir::Value> &nonconstants,
    SmallPtrSetImpl<mlir::Value> &returnvals, ReturnType returnValue,
    DIFFE_TYPE returnType, Twine name, BlockAndValueMapping &VMap,
    std::map<Operation *, Operation *> &OpMap, bool diffeReturnArg,
    mlir::Type additionalArg) {
  assert(!F.getBody().empty());
  // F = preprocessForClone(F, mode);
  // llvm::ValueToValueMapTy VMap;
  auto FTy = getFunctionTypeForClone(
      F.getFunctionType().cast<mlir::FunctionType>(), mode, width,
      additionalArg, constant_args, diffeReturnArg, returnValue, returnType);

  /*
  for (Block &BB : F.getBody().getBlocks()) {
    if (auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      if (auto rv = ri->getReturnValue()) {
        returnvals.insert(rv);
      }
    }
  }
  */

  // Create the new function. This needs to go through the raw Operation API
  // instead of a concrete builder for genericity.
  auto NewF = cast<FunctionOpInterface>(F->cloneWithoutRegions());
  SymbolTable::setSymbolName(NewF, name.str());
  NewF.setType(FTy);

  Operation *parent = F->getParentWithTrait<OpTrait::SymbolTable>();
  SymbolTable table(parent);
  table.insert(NewF);
  SymbolTable::setSymbolVisibility(NewF, SymbolTable::Visibility::Private);

  cloneInto(&F.getBody(), &NewF.getBody(), VMap, OpMap);

  {
    auto &blk = NewF.getBody().front();
    for (ssize_t i = constant_args.size() - 1; i >= 0; i--) {
      mlir::Value oval = F.getBody().front().getArgument(i);
      if (constant_args[i] == DIFFE_TYPE::CONSTANT)
        constants.insert(oval);
      else
        nonconstants.insert(oval);
      if (constant_args[i] == DIFFE_TYPE::DUP_ARG ||
          constant_args[i] == DIFFE_TYPE::DUP_NONEED) {
        mlir::Value val = blk.getArgument(i);
        mlir::Value dval;
        if (i == constant_args.size() - 1)
          dval = blk.addArgument(val.getType(), val.getLoc());
        else
          dval = blk.insertArgument(blk.args_begin() + i + 1, val.getType(),
                                    val.getLoc());
        ptrInputs.map(oval, dval);
      }
    }
  }

  return NewF;
}

class MDiffeGradientUtils : public MGradientUtils {
public:
  MDiffeGradientUtils(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                      FunctionOpInterface oldFunc_, MTypeAnalysis &TA,
                      MTypeResults TR, BlockAndValueMapping &invertedPointers_,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &returnvals_,
                      DIFFE_TYPE ActiveReturn,
                      ArrayRef<DIFFE_TYPE> constant_values,
                      BlockAndValueMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeMode mode, unsigned width, bool omp)
      : MGradientUtils(Logic, newFunc_, oldFunc_, TA, TR, invertedPointers_,
                       constantvalues_, returnvals_, ActiveReturn,
                       constant_values, origToNew_, origToNewOps_, mode, width,
                       omp) {
    /* TODO
    assert(reverseBlocks.size() == 0);
    if (mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeSplit) {
      return;
    }
    for (BasicBlock *BB : originalBlocks) {
      if (BB == inversionAllocs)
        continue;
      BasicBlock *RBB = BasicBlock::Create(BB->getContext(),
                                           "invert" + BB->getName(), newFunc);
      reverseBlocks[BB].push_back(RBB);
      reverseBlockToPrimal[RBB] = BB;
    }
    assert(reverseBlocks.size() != 0);
    */
  }

  // Technically diffe constructor
  static MDiffeGradientUtils *
  CreateFromClone(MEnzymeLogic &Logic, DerivativeMode mode, unsigned width,
                  FunctionOpInterface todiff, MTypeAnalysis &TA,
                  MFnTypeInfo &oldTypeInfo, DIFFE_TYPE retType,
                  bool diffeReturnArg, ArrayRef<DIFFE_TYPE> constant_args,
                  ReturnType returnValue, mlir::Type additionalArg, bool omp) {
    std::string prefix;

    switch (mode) {
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      prefix = "fwddiffe";
      break;
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient:
      prefix = "diffe";
      break;
    case DerivativeMode::ReverseModePrimal:
      llvm_unreachable("invalid DerivativeMode: ReverseModePrimal\n");
    }

    if (width > 1)
      prefix += std::to_string(width);

    BlockAndValueMapping originalToNew;
    std::map<Operation *, Operation *> originalToNewOps;

    SmallPtrSet<mlir::Value, 1> returnvals;
    SmallPtrSet<mlir::Value, 1> constant_values;
    SmallPtrSet<mlir::Value, 1> nonconstant_values;
    BlockAndValueMapping invertedPointers;
    FunctionOpInterface newFunc = CloneFunctionWithReturns(
        mode, width, todiff, invertedPointers, constant_args, constant_values,
        nonconstant_values, returnvals, returnValue, retType,
        prefix + todiff.getName(), originalToNew, originalToNewOps,
        diffeReturnArg, additionalArg);
    MTypeResults TR; // TODO
    return new MDiffeGradientUtils(
        Logic, newFunc, todiff, TA, TR, invertedPointers, constant_values,
        nonconstant_values, retType, constant_args, originalToNew,
        originalToNewOps, mode, width, omp);
  }
};

void createTerminator(MDiffeGradientUtils *gutils, mlir::Block *oBB,
                      DIFFE_TYPE retType, ReturnType retVal) {
  MTypeResults &TR = gutils->TR;
  auto inst = oBB->getTerminator();

  mlir::Block *nBB = gutils->getNewFromOriginal(inst->getBlock());
  assert(nBB);
  auto newInst = nBB->getTerminator();

  OpBuilder nBuilder(inst);
  nBuilder.setInsertionPointToEnd(nBB);

  if (auto binst = dyn_cast<BranchOpInterface>(inst)) {
    // TODO generalize to cloneWithNewBlockArgs interface
    SmallVector<Value> newVals;

    SmallVector<int32_t> segSizes;
    for (size_t i = 0, len = binst.getSuccessorOperands(0)
                                 .getForwardedOperands()
                                 .getBeginOperandIndex();
         i < len; i++)
      newVals.push_back(gutils->getNewFromOriginal(binst->getOperand(i)));
    segSizes.push_back(newVals.size());
    for (size_t i = 0; i < newInst->getNumSuccessors(); i++) {
      size_t cur = newVals.size();
      for (auto op : binst.getSuccessorOperands(i).getForwardedOperands()) {
        newVals.push_back(gutils->getNewFromOriginal(op));
        if (!gutils->isConstantValue(op)) {
          newVals.push_back(gutils->invertPointerM(op, nBuilder));
        }
      }
      cur = newVals.size() - cur;
      segSizes.push_back(cur);
    }

    SmallVector<NamedAttribute> attrs(newInst->getAttrs());
    for (auto &attr : attrs) {
      if (attr.getName() == "operand_segment_sizes")
        attr.setValue(nBuilder.getDenseI32ArrayAttr(segSizes));
    }

    nBB->push_back(newInst->create(
        newInst->getLoc(), newInst->getName(), TypeRange(), newVals, attrs,
        newInst->getSuccessors(), newInst->getNumRegions()));
    gutils->erase(newInst);
    return;
  }

  // In forward mode we only need to update the return value
  if (!inst->hasTrait<OpTrait::ReturnLike>())
    return;

  SmallVector<mlir::Value, 2> retargs;

  switch (retVal) {
  case ReturnType::Return: {
    auto ret = inst->getOperand(0);

    mlir::Value toret;
    if (retType == DIFFE_TYPE::CONSTANT) {
      toret = gutils->getNewFromOriginal(ret);
    } else if (!isa<mlir::FloatType>(ret.getType()) &&
               TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else if (!gutils->isConstantValue(ret)) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else {
      Type retTy = ret.getType().cast<AutoDiffTypeInterface>().getShadowType();
      toret = retTy.cast<AutoDiffTypeInterface>().createNullValue(nBuilder,
                                                                  ret.getLoc());
    }
    retargs.push_back(toret);

    break;
  }
  case ReturnType::TwoReturns: {
    if (retType == DIFFE_TYPE::CONSTANT)
      assert(false && "Invalid return type");
    auto ret = inst->getOperand(0);

    retargs.push_back(gutils->getNewFromOriginal(ret));

    mlir::Value toret;
    if (retType == DIFFE_TYPE::CONSTANT) {
      toret = gutils->getNewFromOriginal(ret);
    } else if (!isa<mlir::FloatType>(ret.getType()) &&
               TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else if (!gutils->isConstantValue(ret)) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else {
      Type retTy = ret.getType().cast<AutoDiffTypeInterface>().getShadowType();
      toret = retTy.cast<AutoDiffTypeInterface>().createNullValue(nBuilder,
                                                                  ret.getLoc());
    }
    retargs.push_back(toret);
    break;
  }
  case ReturnType::Void: {
    break;
  }
  default: {
    llvm::errs() << "Invalid return type: " << to_string(retVal)
                 << "for function: \n"
                 << gutils->newFunc << "\n";
    assert(false && "Invalid return type for function");
    return;
  }
  }

  nBB->push_back(newInst->create(
      newInst->getLoc(), newInst->getName(), TypeRange(), retargs,
      newInst->getAttrs(), newInst->getSuccessors(), newInst->getNumRegions()));
  gutils->erase(newInst);
  return;
}

} // namespace

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

FunctionOpInterface mlir::enzyme::MEnzymeLogic::CreateForwardDiff(
    FunctionOpInterface fn, DIFFE_TYPE retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA, bool returnUsed,
    DerivativeMode mode, bool freeMemory, size_t width, mlir::Type addedType,
    MFnTypeInfo type_args, std::vector<bool> volatile_args, void *augmented) {
  if (fn.getBody().empty()) {
    llvm::errs() << fn << "\n";
    llvm_unreachable("Differentiating empty function");
  }

  MForwardCacheKey tup = {
      fn, retType, constants,
      // std::map<Argument *, bool>(_uncacheable_args.begin(),
      //                           _uncacheable_args.end()),
      returnUsed, mode, static_cast<unsigned>(width), addedType, type_args};

  if (ForwardCachedFunctions.find(tup) != ForwardCachedFunctions.end()) {
    return ForwardCachedFunctions.find(tup)->second;
  }
  bool retActive = retType != DIFFE_TYPE::CONSTANT;
  ReturnType returnValue =
      returnUsed ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                 : (retActive ? ReturnType::Return : ReturnType::Void);
  auto gutils = MDiffeGradientUtils::CreateFromClone(
      *this, mode, width, fn, TA, type_args, retType,
      /*diffeReturnArg*/ false, constants, returnValue, addedType,
      /*omp*/ false);
  ForwardCachedFunctions[tup] = gutils->newFunc;

  insert_or_assign2<MForwardCacheKey, FunctionOpInterface>(
      ForwardCachedFunctions, tup, gutils->newFunc);

  // gutils->FreeMemory = freeMemory;

  const SmallPtrSet<mlir::Block *, 4> guaranteedUnreachable;
  // = getGuaranteedUnreachable(gutils->oldFunc);

  // gutils->forceActiveDetection();
  gutils->forceAugmentedReturns();
  /*

  // TODO populate with actual unnecessaryInstructions once the dependency
  // cycle with activity analysis is removed
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructionsTmp;
  for (auto BB : guaranteedUnreachable) {
    for (auto &I : *BB)
      unnecessaryInstructionsTmp.insert(&I);
  }
  if (mode == DerivativeMode::ForwardModeSplit)
    gutils->computeGuaranteedFrees();

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(
      *gutils->oldFunc, unnecessaryValues, unnecessaryInstructions,
  returnUsed, mode, gutils, TLI, constant_args, guaranteedUnreachable);
  gutils->unnecessaryValuesP = &unnecessaryValues;

  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                  unnecessaryInstructions, gutils, TLI);
                                  */

  for (Block &oBB : gutils->oldFunc.getBody().getBlocks()) {
    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      auto newBB = gutils->getNewFromOriginal(&oBB);

      SmallVector<Operation *, 4> toerase;
      for (auto &I : oBB) {
        toerase.push_back(&I);
      }
      for (auto I : llvm::reverse(toerase)) {
        gutils->eraseIfUnused(I, /*erase*/ true, /*check*/ false);
      }
      OpBuilder builder(gutils->oldFunc.getContext());
      builder.setInsertionPointToEnd(newBB);
      builder.create<LLVM::UnreachableOp>(gutils->oldFunc.getLoc());
      continue;
    }

    auto term = oBB.getTerminator();
    assert(term);

    auto first = oBB.begin();
    auto last = oBB.empty() ? oBB.end() : std::prev(oBB.end());
    for (auto it = first; it != last; ++it) {
      // TODO: propagate errors.
      (void)gutils->visitChild(&*it);
    }

    createTerminator(gutils, &oBB, retType, returnValue);
  }

  // if (mode == DerivativeMode::ForwardModeSplit && augmenteddata)
  //  restoreCache(gutils, augmenteddata->tapeIndices, guaranteedUnreachable);

  // gutils->eraseFictiousPHIs();

  mlir::Block *entry = &gutils->newFunc.getBody().front();

  // cleanupInversionAllocs(gutils, entry);
  // clearFunctionAttributes(gutils->newFunc);

  /*
  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (4)");
  }
  */

  auto nf = gutils->newFunc;
  delete gutils;

  // if (PostOpt)
  //  PPC.optimizeIntermediate(nf);
  // if (EnzymePrint) {
  //  llvm::errs() << nf << "\n";
  //}
  return nf;
}
