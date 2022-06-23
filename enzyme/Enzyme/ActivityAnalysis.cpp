//===- ActivityAnalysis.cpp - Implementation of Activity Analysis ---------===//
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
// This file contains the implementation of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file consists of two mutually recurive
// functions that compute this for values and instructions, respectively.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/IR/InstIterator.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/InlineAsm.h"

#include "ActivityAnalysis.h"
#include "Utils.h"

#if LLVM_VERSION_MAJOR >= 9
#include "llvm/Demangle/Demangle.h"
#endif

#include "FunctionUtils.h"
#include "LibraryFuncs.h"
#include "TypeAnalysis/TBAA.h"

#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;

extern "C" {
cl::opt<bool>
    EnzymePrintActivity("enzyme-print-activity", cl::init(false), cl::Hidden,
                        cl::desc("Print activity analysis algorithm"));

cl::opt<bool> EnzymeNonmarkedGlobalsInactive(
    "enzyme-globals-default-inactive", cl::init(false), cl::Hidden,
    cl::desc("Consider all nonmarked globals to be inactive"));

cl::opt<bool>
    EnzymeEmptyFnInactive("enzyme-emptyfn-inactive", cl::init(false),
                          cl::Hidden,
                          cl::desc("Empty functions are considered inactive"));

cl::opt<bool>
    EnzymeGlobalActivity("enzyme-global-activity", cl::init(false), cl::Hidden,
                         cl::desc("Enable correct global activity analysis"));
}

#include "llvm/IR/InstIterator.h"
#include <map>
#include <set>
#include <unordered_map>

const char *KnownInactiveFunctionsStartingWith[] = {
    "f90io",
    "$ss5print",
#if LLVM_VERSION_MAJOR <= 8
    "_ZN4core3fmt",
    "_ZN3std2io5stdio6_print",
    "_ZNSt7__cxx1112basic_string",
    "_ZNSt7__cxx1118basic_string",
    "_ZNKSt7__cxx1112basic_string",
    "_ZN9__gnu_cxx12__to_xstringINSt7__cxx1112basic_string",
    "_ZNSt12__basic_file",
    "_ZNSt15basic_streambufIcSt11char_traits",
    "_ZNSt13basic_filebufIcSt11char_traits",
    "_ZNSt14basic_ofstreamIcSt11char_traits",
    "_ZNSi4readEPcl",
    "_ZNKSt14basic_ifstreamIcSt11char_traits",
    "_ZNSt14basic_ifstreamIcSt11char_traits",
    "_ZNSo5writeEPKcl",
    "_ZNSt19basic_ostringstreamIcSt11char_traits",
    "_ZStrsIcSt11char_traitsIcESaIcEERSt13basic_istream",
    "_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostream",
    "_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traits",
    "_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traits",
    "_ZNSoD1Ev",
    "_ZNSoC1EPSt15basic_streambufIcSt11char_traits",
    "_ZStlsISt11char_traitsIcEERSt13basic_ostream",
    "_ZSt16__ostream_insert",
    "_ZStlsIwSt11char_traitsIwEERSt13basic_ostream",
    "_ZNSo9_M_insert",
    "_ZNSt13basic_ostream",
    "_ZNSo3put",
    "_ZNKSt5ctypeIcE13_M_widen_init",
    "_ZNSi3get",
    "_ZNSi7getline",
    "_ZNSirsER",
    "_ZNSt7__cxx1115basic_stringbuf",
    "_ZNSi6ignore",
    "_ZNSt8ios_base",
    "_ZNKSt9basic_ios",
    "_ZNSt9basic_ios",
    "_ZStorSt13_Ios_OpenmodeS_",
    "_ZNSt6locale",
    "_ZNKSt6locale4name",
    "_ZStL8__ioinit"
    "_ZNSt9basic_ios",
    "_ZSt4cout",
    "_ZSt3cin",
    "_ZNSi10_M_extract",
    "_ZNSolsE",
    "_ZSt5flush",
    "_ZNSo5flush",
    "_ZSt4endl",
    "_ZNSaIcE",
#endif
};

const char *KnownInactiveFunctionsContains[] = {
    "__enzyme_float", "__enzyme_double", "__enzyme_integer",
    "__enzyme_pointer"};

const std::set<std::string> InactiveGlobals = {
    "ompi_request_null", "ompi_mpi_double", "ompi_mpi_comm_world", "stderr",
    "stdout", "stdin", "_ZSt3cin", "_ZSt4cout", "_ZSt5wcout", "_ZSt4cerr",
    "_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE",
    "_ZTVSt15basic_streambufIcSt11char_traitsIcEE",
    "_ZTVSt9basic_iosIcSt11char_traitsIcEE",
    // istream
    "_ZTVNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE",
    "_ZTTNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE",
    // ostream
    "_ZTVNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE",
    "_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE",
    // stringstream
    "_ZTVNSt7__cxx1118basic_stringstreamIcSt11char_traitsIcESaIcEEE",
    "_ZTTNSt7__cxx1118basic_stringstreamIcSt11char_traitsIcESaIcEEE",
    // vtable for __cxxabiv1::__si_class_type_info
    "_ZTVN10__cxxabiv120__si_class_type_infoE",
    "_ZTVN10__cxxabiv117__class_type_infoE"};

const std::map<std::string, size_t> MPIInactiveCommAllocators = {
    {"MPI_Graph_create", 5},
    {"MPI_Comm_split", 2},
    {"MPI_Intercomm_create", 6},
    {"MPI_Comm_spawn", 6},
    {"MPI_Comm_spawn_multiple", 7},
    {"MPI_Comm_accept", 4},
    {"MPI_Comm_connect", 4},
    {"MPI_Comm_create", 2},
    {"MPI_Comm_create_group", 3},
    {"MPI_Comm_dup", 1},
    {"MPI_Comm_dup", 2},
    {"MPI_Comm_idup", 1},
    {"MPI_Comm_join", 1},
};

const std::set<std::string> KnownInactiveFunctions = {
    "abort",
    "__assert_fail",
    "__cxa_atexit",
    "__cxa_guard_acquire",
    "__cxa_guard_release",
    "__cxa_guard_abort",
    "snprintf",
    "sprintf",
    "printf",
    "fprintf",
    "putchar",
    "fprintf",
    "vprintf",
    "vsnprintf",
    "puts",
    "fflush",
    "__kmpc_for_static_init_4",
    "__kmpc_for_static_init_4u",
    "__kmpc_for_static_init_8",
    "__kmpc_for_static_init_8u",
    "__kmpc_for_static_fini",
    "__kmpc_dispatch_init_4",
    "__kmpc_dispatch_init_4u",
    "__kmpc_dispatch_init_8",
    "__kmpc_dispatch_init_8u",
    "__kmpc_dispatch_next_4",
    "__kmpc_dispatch_next_4u",
    "__kmpc_dispatch_next_8",
    "__kmpc_dispatch_next_8u",
    "__kmpc_dispatch_fini_4",
    "__kmpc_dispatch_fini_4u",
    "__kmpc_dispatch_fini_8",
    "__kmpc_dispatch_fini_8u",
    "__kmpc_barrier",
    "__kmpc_barrier_master",
    "__kmpc_barrier_master_nowait",
    "__kmpc_barrier_end_barrier_master",
    "__kmpc_global_thread_num",
    "omp_get_max_threads",
    "malloc_usable_size",
    "malloc_size",
    "MPI_Init",
    "MPI_Comm_size",
    "PMPI_Comm_size",
    "MPI_Comm_rank",
    "PMPI_Comm_rank",
    "MPI_Get_processor_name",
    "MPI_Finalize",
    "MPI_Test",
    "MPI_Probe", // double check potential syncronization
    "MPI_Barrier",
    "MPI_Abort",
    "MPI_Get_count",
    "MPI_Comm_free",
    "MPI_Comm_get_parent",
    "MPI_Comm_get_name",
    "MPI_Comm_get_info",
    "MPI_Comm_remote_size",
    "MPI_Comm_set_info",
    "MPI_Comm_set_name",
    "MPI_Comm_compare",
    "MPI_Comm_call_errhandler",
    "MPI_Comm_create_errhandler",
    "MPI_Comm_disconnect",
    "MPI_Wtime",
    "_msize",
    "ftnio_fmt_write64",
    "f90_strcmp_klen",
    "__swift_instantiateConcreteTypeFromMangledName",
    "logb",
    "logbf",
    "logbl",
};

const char *DemangledKnownInactiveFunctionsStartingWith[] = {
    "fprintf",
    "std::allocator",
    "std::string",
    "std::cerr",
    "std::istream",
    "std::ostream",
    "std::ios_base",
    "std::locale",
    "std::ctype<char>",
    "std::__basic_file",
    "std::__ioinit",
    "std::__basic_file",

    // __cxx11
    "std::__cxx11::basic_string",
    "std::__cxx11::basic_ios",
    "std::__cxx11::basic_ostringstream",
    "std::__cxx11::basic_istringstream",
    "std::__cxx11::basic_istream",
    "std::__cxx11::basic_ostream",
    "std::__cxx11::basic_ifstream",
    "std::__cxx11::basic_ofstream",
    "std::__cxx11::basic_stringbuf",
    "std::__cxx11::basic_filebuf",
    "std::__cxx11::basic_streambuf",

    // non __cxx11
    "std::basic_string",
    "std::basic_ios",
    "std::basic_ostringstream",
    "std::basic_istringstream",
    "std::basic_istream",
    "std::basic_ostream",
    "std::basic_ifstream",
    "std::basic_ofstream",
    "std::basic_stringbuf",
    "std::basic_filebuf",
    "std::basic_streambuf",

};

/// Is the use of value val as an argument of call CI known to be inactive
/// This tool can only be used when in DOWN mode
bool ActivityAnalyzer::isFunctionArgumentConstant(CallInst *CI, Value *val) {
  assert(directions & DOWN);
  if (CI->hasFnAttr("enzyme_inactive"))
    return true;

  Function *F = getFunctionFromCall(CI);

  // Indirect function calls may actively use the argument
  if (F == nullptr)
    return false;

  if (F->hasFnAttribute("enzyme_inactive")) {
    return true;
  }

  auto Name = F->getName();

  // Allocations, deallocations, and c++ guards don't impact the activity
  // of arguments
  if (isAllocationFunction(*F, TLI) || isDeallocationFunction(*F, TLI))
    return true;
  if (Name == "posix_memalign")
    return true;

#if LLVM_VERSION_MAJOR >= 9
  std::string demangledName = llvm::demangle(Name.str());
  auto dName = StringRef(demangledName);
  for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
    if (dName.startswith(FuncName)) {
      return true;
    }
  }
  if (demangledName == Name.str()) {
    // Either demangeling failed
    // or they are equal but matching failed
    // if (!Name.startswith("llvm."))
    //  llvm::errs() << "matching failed: " << Name.str() << " "
    //               << demangledName << "\n";
  }
#endif
  for (auto FuncName : KnownInactiveFunctionsStartingWith) {
    if (Name.startswith(FuncName)) {
      return true;
    }
  }

  for (auto FuncName : KnownInactiveFunctionsContains) {
    if (Name.contains(FuncName)) {
      return true;
    }
  }
  if (KnownInactiveFunctions.count(Name.str())) {
    return true;
  }

  if (MPIInactiveCommAllocators.find(Name.str()) !=
      MPIInactiveCommAllocators.end()) {
    return true;
  }
  if (F->getIntrinsicID() == Intrinsic::trap)
    return true;

  /// Only the first argument (magnitude) of copysign is active
  if (F->getIntrinsicID() == Intrinsic::copysign &&
      CI->getArgOperand(0) != val) {
    return true;
  }

  /// Use of the value as a non-src/dst in memset/memcpy/memmove is an inactive
  /// use
  if (F->getIntrinsicID() == Intrinsic::memset && CI->getArgOperand(0) != val &&
      CI->getArgOperand(1) != val)
    return true;
  if (F->getIntrinsicID() == Intrinsic::memcpy && CI->getArgOperand(0) != val &&
      CI->getArgOperand(1) != val)
    return true;
  if (F->getIntrinsicID() == Intrinsic::memmove &&
      CI->getArgOperand(0) != val && CI->getArgOperand(1) != val)
    return true;

  // only the float arg input is potentially active
  if (Name == "frexp" || Name == "frexpf" || Name == "frexpl") {
    return val != CI->getOperand(0);
  }

  // The relerr argument is inactive
  if (Name == "Faddeeva_erf" || Name == "Faddeeva_erfc" ||
      Name == "Faddeeva_erfcx" || Name == "Faddeeva_erfi" ||
      Name == "Faddeeva_dawson") {
#if LLVM_VERSION_MAJOR >= 14
    for (size_t i = 0; i < CI->arg_size() - 1; i++)
#else
    for (size_t i = 0; i < CI->getNumArgOperands() - 1; i++)
#endif
    {
      if (val == CI->getOperand(i))
        return false;
    }
    return true;
  }

  // only the buffer is active for mpi send/recv
  if (Name == "MPI_Recv" || Name == "PMPI_Recv" || Name == "MPI_Send" ||
      Name == "PMPI_Send") {
    return val != CI->getOperand(0);
  }
  // only the recv buffer and request is active for mpi isend/irecv
  if (Name == "MPI_Irecv" || Name == "MPI_Isend") {
    return val != CI->getOperand(0) && val != CI->getOperand(6);
  }

  // only request is active
  if (Name == "MPI_Wait" || Name == "PMPI_Wait")
    return val != CI->getOperand(0);

  if (Name == "MPI_Waitall" || Name == "PMPI_Waitall")
    return val != CI->getOperand(1);

  // TODO interprocedural detection
  // Before potential introprocedural detection, any function without definition
  // may to be assumed to have an active use
  if (F->empty())
    return false;

  // With all other options exhausted we have to assume this function could
  // actively use the value
  return false;
}

/// Call the function propagateFromOperand on all operands of CI
/// that could impact the activity of the call instruction
static inline void propagateArgumentInformation(
    TargetLibraryInfo &TLI, CallInst &CI,
    std::function<bool(Value *)> propagateFromOperand) {

  if (auto F = CI.getCalledFunction()) {
    // These functions are known to only have the first argument impact
    // the activity of the call instruction
    auto Name = F->getName();
    if (Name == "lgamma" || Name == "lgammaf" || Name == "lgammal" ||
        Name == "lgamma_r" || Name == "lgammaf_r" || Name == "lgammal_r" ||
        Name == "__lgamma_r_finite" || Name == "__lgammaf_r_finite" ||
        Name == "__lgammal_r_finite") {

      propagateFromOperand(CI.getArgOperand(0));
      return;
    }

    // Allocations, deallocations, and c++ guards are fully inactive
    if (isAllocationFunction(*F, TLI) || isDeallocationFunction(*F, TLI) ||
        Name == "__cxa_guard_acquire" || Name == "__cxa_guard_release" ||
        Name == "__cxa_guard_abort")
      return;

    /// Only the first argument (magnitude) of copysign is active
    if (F->getIntrinsicID() == Intrinsic::copysign) {
      propagateFromOperand(CI.getOperand(0));
      return;
    }

    /// Only the src/dst in memset/memcpy/memmove impact the activity of the
    /// instruction
    // memset cannot propagate activity as it sets
    // data to a given single byte which is inactive
    if (F->getIntrinsicID() == Intrinsic::memset) {
      return;
    }
    if (F->getIntrinsicID() == Intrinsic::memcpy ||
        F->getIntrinsicID() == Intrinsic::memmove) {
      propagateFromOperand(CI.getOperand(0));
      propagateFromOperand(CI.getOperand(1));
      return;
    }

    if (Name == "frexp" || Name == "frexpf" || Name == "frexpl") {
      propagateFromOperand(CI.getOperand(0));
      return;
    }
    if (Name == "Faddeeva_erf" || Name == "Faddeeva_erfc" ||
        Name == "Faddeeva_erfcx" || Name == "Faddeeva_erfi" ||
        Name == "Faddeeva_dawson") {
#if LLVM_VERSION_MAJOR >= 14
      for (size_t i = 0; i < CI.arg_size() - 1; i++)
#else
      for (size_t i = 0; i < CI.getNumArgOperands() - 1; i++)
#endif
      {
        propagateFromOperand(CI.getOperand(i));
      }
      return;
    }
  }

  // For other calls, check all operands of the instruction
  // as conservatively they may impact the activity of the call
#if LLVM_VERSION_MAJOR >= 14
  for (auto &a : CI.args())
#else
  for (auto &a : CI.arg_operands())
#endif
  {
    if (propagateFromOperand(a))
      break;
  }
}

/// Return whether this instruction is known not to propagate adjoints
/// Note that instructions could return an active pointer, but
/// do not propagate adjoints themselves
bool ActivityAnalyzer::isConstantInstruction(TypeResults const &TR,
                                             Instruction *I) {
  // This analysis may only be called by instructions corresponding to
  // the function analyzed by TypeInfo
  assert(I);
  assert(TR.getFunction() == I->getParent()->getParent());

  // The return instruction doesn't impact activity (handled specifically
  // during adjoint generation)
  if (isa<ReturnInst>(I))
    return true;

  // Branch, unreachable, and previously computed constants are inactive
  if (isa<UnreachableInst>(I) || isa<BranchInst>(I) ||
      (ConstantInstructions.find(I) != ConstantInstructions.end())) {
    return true;
  }

  /// Previously computed inactives remain inactive
  if ((ActiveInstructions.find(I) != ActiveInstructions.end())) {
    return false;
  }

  if (notForAnalysis.count(I->getParent())) {
    if (EnzymePrintActivity)
      llvm::errs() << " constant instruction as dominates unreachable " << *I
                   << "\n";
    InsertConstantInstruction(TR, I);
    return true;
  }

  if (auto CI = dyn_cast<CallInst>(I)) {
    if (CI->hasFnAttr("enzyme_active")) {
      if (EnzymePrintActivity)
        llvm::errs() << "forced active " << *I << "\n";
      ActiveInstructions.insert(I);
      return false;
    }
    if (CI->hasFnAttr("enzyme_inactive")) {
      if (EnzymePrintActivity)
        llvm::errs() << "forced inactive " << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }
    Function *called = getFunctionFromCall(CI);

    if (called) {
      if (called->hasFnAttribute("enzyme_active")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced active " << *I << "\n";
        ActiveInstructions.insert(I);
        return false;
      }
      if (called->hasFnAttribute("enzyme_inactive")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced inactive " << *I << "\n";
        InsertConstantInstruction(TR, I);
        return true;
      }
    }
  }

  /// A store into all integral memory is inactive
  if (auto SI = dyn_cast<StoreInst>(I)) {
    auto StoreSize = SI->getParent()
                         ->getParent()
                         ->getParent()
                         ->getDataLayout()
                         .getTypeSizeInBits(SI->getValueOperand()->getType()) /
                     8;

    bool AllIntegral = true;
    bool SeenInteger = false;
    auto q = TR.query(SI->getPointerOperand()).Data0();
    for (int i = -1; i < (int)StoreSize; ++i) {
      auto dt = q[{i}];
      if (dt.isIntegral() || dt == BaseType::Anything) {
        SeenInteger = true;
        if (i == -1)
          break;
      } else if (dt.isKnown()) {
        AllIntegral = false;
        break;
      }
    }

    if (AllIntegral && SeenInteger) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction from TA " << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }
  }
  if (auto SI = dyn_cast<AtomicRMWInst>(I)) {
    auto StoreSize = SI->getParent()
                         ->getParent()
                         ->getParent()
                         ->getDataLayout()
                         .getTypeSizeInBits(I->getType()) /
                     8;

    bool AllIntegral = true;
    bool SeenInteger = false;
    auto q = TR.query(SI->getOperand(0)).Data0();
    for (int i = -1; i < (int)StoreSize; ++i) {
      auto dt = q[{i}];
      if (dt.isIntegral() || dt == BaseType::Anything) {
        SeenInteger = true;
        if (i == -1)
          break;
      } else if (dt.isKnown()) {
        AllIntegral = false;
        break;
      }
    }

    if (AllIntegral && SeenInteger) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction from TA " << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }
  }

  if (isa<MemSetInst>(I)) {
    // memset's are definitionally inactive since
    // they copy a byte which cannot be active
    if (EnzymePrintActivity)
      llvm::errs() << " constant instruction as memset " << *I << "\n";
    InsertConstantInstruction(TR, I);
    return true;
  }

  if (EnzymePrintActivity)
    llvm::errs() << "checking if is constant[" << (int)directions << "] " << *I
                 << "\n";

  if (auto II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::nvvm_barrier0:
    case Intrinsic::nvvm_barrier0_popc:
    case Intrinsic::nvvm_barrier0_and:
    case Intrinsic::nvvm_barrier0_or:
    case Intrinsic::nvvm_membar_cta:
    case Intrinsic::nvvm_membar_gl:
    case Intrinsic::nvvm_membar_sys:
    case Intrinsic::amdgcn_s_barrier:
    case Intrinsic::assume:
    case Intrinsic::stacksave:
    case Intrinsic::stackrestore:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::dbg_addr:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::var_annotation:
    case Intrinsic::ptr_annotation:
    case Intrinsic::annotation:
    case Intrinsic::codeview_annotation:
    case Intrinsic::expect:
    case Intrinsic::type_test:
    case Intrinsic::donothing:
    case Intrinsic::prefetch:
    case Intrinsic::trap:
#if LLVM_VERSION_MAJOR >= 8
    case Intrinsic::is_constant:
#endif
      if (EnzymePrintActivity)
        llvm::errs() << "known inactive intrinsic " << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;

    default:
      break;
    }
  }

  // Analyzer for inductive assumption where we attempt to prove this is
  // inactive from a lack of active users
  std::shared_ptr<ActivityAnalyzer> DownHypothesis;

  // If this instruction does not write to memory that outlives itself
  // (potentially propagating derivative information), the only way to propagate
  // derivative information is through the return value
  // TODO the "doesn't write to active memory" can be made more aggressive than
  // doesn't write to any memory
  bool noActiveWrite = false;
  if (!I->mayWriteToMemory())
    noActiveWrite = true;
  else if (auto CI = dyn_cast<CallInst>(I)) {
    if (AA.onlyReadsMemory(CI)) {
      noActiveWrite = true;
    } else if (auto F = CI->getCalledFunction()) {
      if (isMemFreeLibMFunction(F->getName())) {
        noActiveWrite = true;
      } else if (F->getName() == "frexp" || F->getName() == "frexpf" ||
                 F->getName() == "frexpl") {
        noActiveWrite = true;
      }
    }
  }
  if (noActiveWrite) {
    // Even if returning a pointer, this instruction is considered inactive
    // since the instruction doesn't prop gradients. Thus, so long as we don't
    // return an object containing a float, this instruction is inactive
    if (!TR.intType(1, I, /*errifNotFound*/ false).isPossibleFloat()) {
      if (EnzymePrintActivity)
        llvm::errs()
            << " constant instruction from known non-float non-writing "
               "instruction "
            << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }

    // If the value returned is constant otherwise, the instruction is inactive
    if (isConstantValue(TR, I)) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction from known constant non-writing "
                        "instruction "
                     << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }

    // Even if the return is nonconstant, it's worth checking explicitly the
    // users since unlike isConstantValue, returning a pointer does not make the
    // instruction active
    if (directions & DOWN) {
      // We shall now induct on this instruction being inactive and try to prove
      // this fact from a lack of active users.

      // If we aren't a phi node (and thus potentially recursive on uses) and
      // already equal to the current direction, we don't need to induct,
      // reducing runtime.
      if (directions == DOWN && !isa<PHINode>(I)) {
        if (isValueInactiveFromUsers(TR, I, UseActivity::None)) {
          if (EnzymePrintActivity)
            llvm::errs() << " constant instruction[" << (int)directions
                         << "] from users instruction " << *I << "\n";
          InsertConstantInstruction(TR, I);
          return true;
        }
      } else {
        DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
            new ActivityAnalyzer(*this, DOWN));
        DownHypothesis->ConstantInstructions.insert(I);
        if (DownHypothesis->isValueInactiveFromUsers(TR, I,
                                                     UseActivity::None)) {
          if (EnzymePrintActivity)
            llvm::errs() << " constant instruction[" << (int)directions
                         << "] from users instruction " << *I << "\n";
          InsertConstantInstruction(TR, I);
          insertConstantsFrom(TR, *DownHypothesis);
          return true;
        }
      }
    }
  }

  std::shared_ptr<ActivityAnalyzer> UpHypothesis;
  if (directions & UP) {
    // If this instruction has no active operands, the instruction
    // is active.
    // TODO This isn't 100% accurate and will incorrectly mark a no-argument
    // function that reads from active memory as constant
    // Technically the additional constraint is that this does not read from
    // active memory, where we have assumed that the only active memory
    // we care about is accessible from arguments passed (and thus not globals)
    UpHypothesis =
        std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
    UpHypothesis->ConstantInstructions.insert(I);
    assert(directions & UP);
    if (UpHypothesis->isInstructionInactiveFromOrigin(TR, I)) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction from origin "
                        "instruction "
                     << *I << "\n";
      InsertConstantInstruction(TR, I);
      insertConstantsFrom(TR, *UpHypothesis);
      if (DownHypothesis)
        insertConstantsFrom(TR, *DownHypothesis);
      return true;
    } else if (directions == 3) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<BinaryOperator>(I)) {
        for (auto &op : I->operands()) {
          if (!UpHypothesis->isConstantValue(TR, op)) {
            ReEvaluateInstIfInactiveValue[op].insert(I);
          }
        }
      }
    }
  }

  // Otherwise we must fall back and assume this instruction to be active.
  ActiveInstructions.insert(I);
  if (EnzymePrintActivity)
    llvm::errs() << "couldnt decide fallback as nonconstant instruction("
                 << (int)directions << "):" << *I << "\n";
  if (noActiveWrite && directions == 3)
    ReEvaluateInstIfInactiveValue[I].insert(I);
  return false;
}

bool isValuePotentiallyUsedAsPointer(llvm::Value *val) {
  std::deque<llvm::Value *> todo = {val};
  SmallPtrSet<Value *, 3> seen;
  while (todo.size()) {
    auto cur = todo.back();
    todo.pop_back();
    if (seen.count(cur))
      continue;
    seen.insert(cur);
    for (auto u : cur->users()) {
      if (isa<ReturnInst>(u))
        return true;
      if (!cast<Instruction>(u)->mayReadOrWriteMemory()) {
        todo.push_back(u);
        continue;
      }
      if (EnzymePrintActivity)
        llvm::errs() << " VALUE potentially used as pointer " << *val << " by "
                     << *u << "\n";
      return true;
    }
  }
  return false;
}

bool ActivityAnalyzer::isConstantValue(TypeResults const &TR, Value *Val) {
  // This analysis may only be called by instructions corresponding to
  // the function analyzed by TypeInfo -- however if the Value
  // was created outside a function (e.g. global, constant), that is allowed
  assert(Val);
  if (auto I = dyn_cast<Instruction>(Val)) {
    if (TR.getFunction() != I->getParent()->getParent()) {
      llvm::errs() << *TR.getFunction() << "\n";
      llvm::errs() << *I << "\n";
    }
    assert(TR.getFunction() == I->getParent()->getParent());
  }
  if (auto Arg = dyn_cast<Argument>(Val)) {
    assert(TR.getFunction() == Arg->getParent());
  }

  // Void values are definitionally inactive
  if (Val->getType()->isVoidTy())
    return true;

  // Token values are definitionally inactive
  if (Val->getType()->isTokenTy())
    return true;

  // All function pointers are considered active in case an augmented primal
  // or reverse is needed
  if (isa<Function>(Val) || isa<InlineAsm>(Val)) {
    return false;
  }

  /// If we've already shown this value to be inactive
  if (ConstantValues.find(Val) != ConstantValues.end()) {
    return true;
  }

  /// If we've already shown this value to be active
  if (ActiveValues.find(Val) != ActiveValues.end()) {
    return false;
  }

  if (auto CD = dyn_cast<ConstantDataSequential>(Val)) {
    // inductively assume inactive
    ConstantValues.insert(CD);
    for (size_t i = 0, len = CD->getNumElements(); i < len; i++) {
      if (!isConstantValue(TR, CD->getElementAsConstant(i))) {
        ConstantValues.erase(CD);
        ActiveValues.insert(CD);
        return false;
      }
    }
    return true;
  }
  if (auto CD = dyn_cast<ConstantAggregate>(Val)) {
    // inductively assume inactive
    ConstantValues.insert(CD);
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      if (!isConstantValue(TR, CD->getOperand(i))) {
        ConstantValues.erase(CD);
        ActiveValues.insert(CD);
        return false;
      }
    }
    return true;
  }

  // Undef, metadata, non-global constants, and blocks are inactive
  if (isa<UndefValue>(Val) || isa<MetadataAsValue>(Val) ||
      isa<ConstantData>(Val) || isa<ConstantAggregate>(Val) ||
      isa<BasicBlock>(Val)) {
    return true;
  }

  if (auto II = dyn_cast<IntrinsicInst>(Val)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::nvvm_barrier0:
    case Intrinsic::nvvm_barrier0_popc:
    case Intrinsic::nvvm_barrier0_and:
    case Intrinsic::nvvm_barrier0_or:
    case Intrinsic::nvvm_membar_cta:
    case Intrinsic::nvvm_membar_gl:
    case Intrinsic::nvvm_membar_sys:
    case Intrinsic::amdgcn_s_barrier:
    case Intrinsic::assume:
    case Intrinsic::stacksave:
    case Intrinsic::stackrestore:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::dbg_addr:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::var_annotation:
    case Intrinsic::ptr_annotation:
    case Intrinsic::annotation:
    case Intrinsic::codeview_annotation:
    case Intrinsic::expect:
    case Intrinsic::type_test:
    case Intrinsic::donothing:
    case Intrinsic::prefetch:
#if LLVM_VERSION_MAJOR >= 8
    case Intrinsic::is_constant:
#endif
      InsertConstantValue(TR, Val);
      return true;
    default:
      break;
    }
  }

  // All arguments must be marked constant/nonconstant ahead of time
  if (isa<Argument>(Val) && !cast<Argument>(Val)->hasByValAttr()) {
    llvm::errs() << *(cast<Argument>(Val)->getParent()) << "\n";
    llvm::errs() << *Val << "\n";
    assert(0 && "must've put arguments in constant/nonconstant");
  }

  // This value is certainly an integer (and only and integer, not a pointer or
  // float). Therefore its value is constant
  if (TR.intType(1, Val, /*errIfNotFound*/ false).isIntegral()) {
    if (EnzymePrintActivity)
      llvm::errs() << " Value const as integral " << (int)directions << " "
                   << *Val << " "
                   << TR.intType(1, Val, /*errIfNotFound*/ false).str() << "\n";
    InsertConstantValue(TR, Val);
    return true;
  }

#if 0
  // This value is certainly a pointer to an integer (and only and integer, not
  // a pointer or float). Therefore its value is constant
  // TODO use typeInfo for more aggressive activity analysis
  if (val->getType()->isPointerTy() &&
      cast<PointerType>(val->getType())->isIntOrIntVectorTy() &&
      TR.firstPointer(1, val, /*errifnotfound*/ false).isIntegral()) {
    if (EnzymePrintActivity)
      llvm::errs() << " Value const as integral pointer" << (int)directions
                   << " " << *val << "\n";
    InsertConstantValue(TR, Val);
    return true;
  }
#endif

  if (auto GI = dyn_cast<GlobalVariable>(Val)) {
    // If operating under the assumption globals are inactive unless
    // explicitly marked as active, this is inactive
    if (!hasMetadata(GI, "enzyme_shadow") && EnzymeNonmarkedGlobalsInactive) {
      InsertConstantValue(TR, Val);
      return true;
    }

    if (GI->getName().contains("enzyme_const") ||
        InactiveGlobals.count(GI->getName().str())) {
      InsertConstantValue(TR, Val);
      return true;
    }

    // If this global is unchanging and the internal constant data
    // is inactive, the global is inactive
    if (GI->isConstant() && GI->hasInitializer() &&
        isConstantValue(TR, GI->getInitializer())) {
      InsertConstantValue(TR, Val);
      if (EnzymePrintActivity)
        llvm::errs() << " VALUE const global " << *Val
                     << " init: " << *GI->getInitializer() << "\n";
      return true;
    }

    // If this global is a pointer to an integer, it is inactive
    // TODO note this may need updating to consider the size
    // of the global
    auto res = TR.query(GI).Data0();
    auto dt = res[{-1}];
    if (dt.isIntegral()) {
      if (EnzymePrintActivity)
        llvm::errs() << " VALUE const as global int pointer " << *Val
                     << " type - " << res.str() << "\n";
      InsertConstantValue(TR, Val);
      return true;
    }

    // If this is a global local to this translation unit with inactive
    // initializer and no active uses, it is definitionally inactive
    bool usedJustInThisModule =
        GI->hasInternalLinkage() || GI->hasPrivateLinkage();

    if (EnzymePrintActivity)
      llvm::errs() << "pre attempting(" << (int)directions
                   << ") just used in module for: " << *GI << " dir"
                   << (int)directions << " justusedin:" << usedJustInThisModule
                   << "\n";

    if (directions == 3 && usedJustInThisModule) {
      // TODO this assumes global initializer cannot refer to itself (lest
      // infinite loop)
      if (!GI->hasInitializer() || isConstantValue(TR, GI->getInitializer())) {

        if (EnzymePrintActivity)
          llvm::errs() << "attempting just used in module for: " << *GI << "\n";
        // Not looking at users to prove inactive (definition of down)
        // If all users are inactive, this is therefore inactive.
        // Since we won't look at origins to prove, we can inductively assume
        // this is inactive

        // As an optimization if we are going down already
        // and we won't use ourselves (done by PHI's), we
        // dont need to inductively assume we're true
        // and can instead use this object!
        // This pointer is inactive if it is either not actively stored to or
        // not actively loaded from
        // See alloca logic to explain why OnlyStores is insufficient here
        if (directions == DOWN) {
          if (isValueInactiveFromUsers(TR, Val, UseActivity::OnlyLoads)) {
            InsertConstantValue(TR, Val);
            return true;
          }
        } else {
          Instruction *LoadReval = nullptr;
          Instruction *StoreReval = nullptr;
          auto DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
              new ActivityAnalyzer(*this, DOWN));
          DownHypothesis->ConstantValues.insert(Val);
          if (DownHypothesis->isValueInactiveFromUsers(
                  TR, Val, UseActivity::OnlyLoads, &LoadReval) ||
              (TR.query(GI)[{-1, -1}].isFloat() &&
               DownHypothesis->isValueInactiveFromUsers(
                   TR, Val, UseActivity::OnlyStores, &StoreReval))) {
            insertConstantsFrom(TR, *DownHypothesis);
            InsertConstantValue(TR, Val);
            return true;
          } else {
            if (LoadReval) {
              if (EnzymePrintActivity)
                llvm::errs() << " global activity of " << *Val
                             << " dependant on " << *LoadReval << "\n";
              ReEvaluateValueIfInactiveInst[LoadReval].insert(Val);
            }
            if (StoreReval)
              ReEvaluateValueIfInactiveInst[StoreReval].insert(Val);
          }
        }
      }
    }

    // Otherwise we have to assume this global is active since it can
    // be arbitrarily used in an active way
    // TODO we can be more aggressive here in the future
    if (EnzymePrintActivity)
      llvm::errs() << " VALUE nonconst unknown global " << *Val << " type - "
                   << res.str() << "\n";
    ActiveValues.insert(Val);
    return false;
  }

  // ConstantExpr's are inactive if their arguments are inactive
  // Note that since there can't be a recursive constant this shouldn't
  // infinite loop
  if (auto ce = dyn_cast<ConstantExpr>(Val)) {
    if (ce->isCast()) {
      if (auto PT = dyn_cast<PointerType>(ce->getType())) {
        if (PT->getPointerElementType()->isFunctionTy()) {
          if (EnzymePrintActivity)
            llvm::errs()
                << " VALUE nonconst as cast to pointer of functiontype " << *Val
                << "\n";
          ActiveValues.insert(Val);
          return false;
        }
      }

      if (isConstantValue(TR, ce->getOperand(0))) {
        if (EnzymePrintActivity)
          llvm::errs() << " VALUE const cast from from operand " << *Val
                       << "\n";
        InsertConstantValue(TR, Val);
        return true;
      }
    }
    if (ce->getOpcode() == Instruction::GetElementPtr &&
        llvm::all_of(ce->operand_values(),
                     [&](Value *v) { return isConstantValue(TR, v); })) {
      if (isConstantValue(TR, ce->getOperand(0))) {
        if (EnzymePrintActivity)
          llvm::errs() << " VALUE const cast from gep operand " << *Val << "\n";
        InsertConstantValue(TR, Val);
        return true;
      }
    }
    if (EnzymePrintActivity)
      llvm::errs() << " VALUE nonconst unknown expr " << *Val << "\n";
    ActiveValues.insert(Val);
    return false;
  }

  if (auto CI = dyn_cast<CallInst>(Val)) {
    if (CI->hasFnAttr("enzyme_active")) {
      if (EnzymePrintActivity)
        llvm::errs() << "forced active val " << *Val << "\n";
      ActiveValues.insert(Val);
      return false;
    }
    if (CI->hasFnAttr("enzyme_inactive")) {
      if (EnzymePrintActivity)
        llvm::errs() << "forced inactive val " << *Val << "\n";
      InsertConstantValue(TR, Val);
      return true;
    }
    Function *called = getFunctionFromCall(CI);

    if (called) {
      if (called->hasFnAttribute("enzyme_active")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced active val " << *Val << "\n";
        ActiveValues.insert(Val);
        return false;
      }
      if (called->hasFnAttribute("enzyme_inactive")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced inactive val " << *Val << "\n";
        InsertConstantValue(TR, Val);
        return true;
      }
    }
  }

  std::shared_ptr<ActivityAnalyzer> UpHypothesis;

  // Handle types that could contain pointers
  //  Consider all types except
  //   * floating point types (since those are assumed not pointers)
  //   * integers that we know are not pointers
  bool containsPointer = true;
  if (Val->getType()->isFPOrFPVectorTy())
    containsPointer = false;
  if (!TR.intType(1, Val, /*errIfNotFound*/ false).isPossiblePointer())
    containsPointer = false;

  if (containsPointer && !isValuePotentiallyUsedAsPointer(Val)) {
    containsPointer = false;
  }

  if (containsPointer) {

    auto TmpOrig =
#if LLVM_VERSION_MAJOR >= 12
        getUnderlyingObject(Val, 100);
#else
        GetUnderlyingObject(Val, TR.getFunction()->getParent()->getDataLayout(),
                            100);
#endif

    // If we know that our origin is inactive from its arguments,
    // we are definitionally inactive
    if (directions & UP) {
      // If we are derived from an argument our activity is equal to the
      // activity of the argument by definition
      if (auto arg = dyn_cast<Argument>(TmpOrig)) {
        if (!arg->hasByValAttr()) {
          bool res = isConstantValue(TR, TmpOrig);
          if (res) {
            if (EnzymePrintActivity)
              llvm::errs() << " arg const from orig val=" << *Val
                           << " orig=" << *TmpOrig << "\n";
            InsertConstantValue(TR, Val);
          } else {
            if (EnzymePrintActivity)
              llvm::errs() << " arg active from orig val=" << *Val
                           << " orig=" << *TmpOrig << "\n";
            ActiveValues.insert(Val);
          }
          return res;
        }
      }

      UpHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
      UpHypothesis->ConstantValues.insert(Val);

      // If our origin is a load of a known inactive (say inactive argument), we
      // are also inactive
      if (auto PN = dyn_cast<PHINode>(TmpOrig)) {
        // Not taking fast path incase phi is recursive.
        Value *active = nullptr;
        for (auto &V : PN->incoming_values()) {
          if (!UpHypothesis->isConstantValue(TR, V.get())) {
            active = V.get();
            break;
          }
        }
        if (!active) {
          InsertConstantValue(TR, Val);
          if (TmpOrig != Val) {
            InsertConstantValue(TR, TmpOrig);
          }
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        } else {
          ReEvaluateValueIfInactiveValue[active].insert(Val);
          if (TmpOrig != Val) {
            ReEvaluateValueIfInactiveValue[active].insert(TmpOrig);
          }
        }
      } else if (auto LI = dyn_cast<LoadInst>(TmpOrig)) {

        if (directions == UP) {
          if (isConstantValue(TR, LI->getPointerOperand())) {
            InsertConstantValue(TR, Val);
            return true;
          }
        } else {
          if (UpHypothesis->isConstantValue(TR, LI->getPointerOperand())) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
        }
      } else if (isa<IntrinsicInst>(TmpOrig) &&
                 (cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldu_global_i ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldu_global_p ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldu_global_f ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldg_global_i ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldg_global_p ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldg_global_f)) {
        auto II = cast<IntrinsicInst>(TmpOrig);
        if (directions == UP) {
          if (isConstantValue(TR, II->getOperand(0))) {
            InsertConstantValue(TR, Val);
            return true;
          }
        } else {
          if (UpHypothesis->isConstantValue(TR, II->getOperand(0))) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
        }
      } else if (auto op = dyn_cast<CallInst>(TmpOrig)) {
        if (op->hasFnAttr("enzyme_inactive")) {
          InsertConstantValue(TR, Val);
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        }
        Function *called = getFunctionFromCall(op);

        if (called) {
          if (called->hasFnAttribute("enzyme_inactive")) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
          if (called->getName() == "free" || called->getName() == "_ZdlPv" ||
              called->getName() == "_ZdlPvm" || called->getName() == "munmap") {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }

#if LLVM_VERSION_MAJOR >= 9
          auto dName = demangle(called->getName().str());
          for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
            if (StringRef(dName).startswith(FuncName)) {
              InsertConstantValue(TR, Val);
              insertConstantsFrom(TR, *UpHypothesis);
              return true;
            }
          }
#endif

          for (auto FuncName : KnownInactiveFunctionsStartingWith) {
            if (called->getName().startswith(FuncName)) {
              InsertConstantValue(TR, Val);
              insertConstantsFrom(TR, *UpHypothesis);
              return true;
            }
          }

          for (auto FuncName : KnownInactiveFunctionsContains) {
            if (called->getName().contains(FuncName)) {
              InsertConstantValue(TR, Val);
              insertConstantsFrom(TR, *UpHypothesis);
              return true;
            }
          }

          if (KnownInactiveFunctions.count(called->getName().str()) ||
              MPIInactiveCommAllocators.find(called->getName().str()) !=
                  MPIInactiveCommAllocators.end()) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }

          if (called->getIntrinsicID() == Intrinsic::trap) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }

          // If requesting empty unknown functions to be considered inactive,
          // abide by those rules
          if (EnzymeEmptyFnInactive && called->empty() &&
              !hasMetadata(called, "enzyme_gradient") &&
              !hasMetadata(called, "enzyme_derivative") &&
              !isAllocationFunction(*called, TLI) &&
              !isDeallocationFunction(*called, TLI) &&
              !isa<IntrinsicInst>(op)) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
        }
      } else if (isa<AllocaInst>(Val)) {
        // This pointer is inactive if it is either not actively stored to or
        // not actively loaded from and is nonescaping by definition of being
        // alloca OnlyStores is insufficient here since the loaded pointer can
        // have active memory stored into it [e.g. not just top level pointer
        // that matters]
        if (directions == DOWN) {
          if (isValueInactiveFromUsers(TR, TmpOrig, UseActivity::OnlyLoads)) {
            InsertConstantValue(TR, Val);
            return true;
          }
        } else if (directions & DOWN) {
          Instruction *LoadReval = nullptr;
          auto DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
              new ActivityAnalyzer(*this, DOWN));
          DownHypothesis->ConstantValues.insert(TmpOrig);
          if (DownHypothesis->isValueInactiveFromUsers(
                  TR, TmpOrig, UseActivity::OnlyLoads, &LoadReval)) {
            insertConstantsFrom(TR, *DownHypothesis);
            InsertConstantValue(TR, Val);
            return true;
          } else {
            if (LoadReval) {
              ReEvaluateValueIfInactiveInst[LoadReval].insert(TmpOrig);
            }
          }
        }
      }

      // otherwise if the origin is a previously derived known inactive value
      // assess
      // TODO here we would need to potentially consider loading an active
      // global as we again assume that active memory is passed explicitly as an
      // argument
      if (TmpOrig != Val) {
        if (isConstantValue(TR, TmpOrig)) {
          if (EnzymePrintActivity)
            llvm::errs() << " Potential Pointer(" << (int)directions << ") "
                         << *Val << " inactive from inactive origin "
                         << *TmpOrig << "\n";
          InsertConstantValue(TR, Val);
          return true;
        }
      }
      if (auto inst = dyn_cast<Instruction>(Val)) {
        if (!inst->mayReadFromMemory() && !isa<AllocaInst>(Val)) {
          if (directions == UP && !isa<PHINode>(inst)) {
            if (isInstructionInactiveFromOrigin(TR, inst)) {
              InsertConstantValue(TR, Val);
              return true;
            }
          } else {
            if (UpHypothesis->isInstructionInactiveFromOrigin(TR, inst)) {
              InsertConstantValue(TR, Val);
              insertConstantsFrom(TR, *UpHypothesis);
              return true;
            }
          }
        }
      }
    }

    // If not capable of looking at both users and uses, all the ways a pointer
    // can be loaded/stored cannot be assesed and therefore we default to assume
    // it to be active
    if (directions != 3) {
      if (EnzymePrintActivity)
        llvm::errs() << " <Potential Pointer assumed active at "
                     << (int)directions << ">" << *Val << "\n";
      ActiveValues.insert(Val);
      return false;
    }

    if (EnzymePrintActivity)
      llvm::errs() << " < MEMSEARCH" << (int)directions << ">" << *Val << "\n";
    // A pointer value is active if two things hold:
    //   an potentially active value is stored into the memory
    //   memory loaded from the value is used in an active way
    bool potentiallyActiveStore = false;
    bool potentialStore = false;
    bool potentiallyActiveLoad = false;

    // Assume the value (not instruction) is itself active
    // In spite of that can we show that there are either no active stores
    // or no active loads
    std::shared_ptr<ActivityAnalyzer> Hypothesis =
        std::shared_ptr<ActivityAnalyzer>(
            new ActivityAnalyzer(*this, directions));
    Hypothesis->ActiveValues.insert(Val);
    if (auto VI = dyn_cast<Instruction>(Val)) {
      for (auto V : DeducingPointers) {
        UpHypothesis->InsertConstantValue(TR, V);
      }
      if (UpHypothesis->isInstructionInactiveFromOrigin(TR, VI)) {
        Hypothesis->DeducingPointers.insert(Val);
        if (EnzymePrintActivity)
          llvm::errs() << " constant instruction hypothesis: " << *VI << "\n";
      } else {
        if (EnzymePrintActivity)
          llvm::errs() << " cannot show constant instruction hypothesis: "
                       << *VI << "\n";
      }
    }

    auto checkActivity = [&](Instruction *I) {
      if (notForAnalysis.count(I->getParent()))
        return false;

      // If this is a malloc or free, this doesn't impact the activity
      if (auto CI = dyn_cast<CallInst>(I)) {
        if (CI->hasFnAttr("enzyme_inactive"))
          return false;

#if LLVM_VERSION_MAJOR >= 11
        if (auto iasm = dyn_cast<InlineAsm>(CI->getCalledOperand()))
#else
        if (auto iasm = dyn_cast<InlineAsm>(CI->getCalledValue()))
#endif
        {
          if (StringRef(iasm->getAsmString()).contains("exit") ||
              StringRef(iasm->getAsmString()).contains("cpuid"))
            return false;
        }

        Function *F = getFunctionFromCall(CI);

        if (F) {
          if (F->hasFnAttribute("enzyme_inactive")) {
            return false;
          }
          if (isAllocationFunction(*F, TLI) ||
              isDeallocationFunction(*F, TLI)) {
            return false;
          }
          if (KnownInactiveFunctions.count(F->getName().str()) ||
              MPIInactiveCommAllocators.find(F->getName().str()) !=
                  MPIInactiveCommAllocators.end()) {
            return false;
          }
          if (isMemFreeLibMFunction(F->getName()) ||
              F->getName() == "__fd_sincos_1") {
            return false;
          }
#if LLVM_VERSION_MAJOR >= 9
          auto dName = demangle(F->getName().str());
          for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
            if (StringRef(dName).startswith(FuncName)) {
              return false;
            }
          }
#endif
          for (auto FuncName : KnownInactiveFunctionsStartingWith) {
            if (F->getName().startswith(FuncName)) {
              return false;
            }
          }
          for (auto FuncName : KnownInactiveFunctionsContains) {
            if (F->getName().contains(FuncName)) {
              return false;
            }
          }

          if (F->getName() == "__cxa_guard_acquire" ||
              F->getName() == "__cxa_guard_release" ||
              F->getName() == "__cxa_guard_abort" ||
              F->getName() == "posix_memalign") {
            return false;
          }

          bool noUse = false;
          switch (F->getIntrinsicID()) {
          case Intrinsic::nvvm_barrier0:
          case Intrinsic::nvvm_barrier0_popc:
          case Intrinsic::nvvm_barrier0_and:
          case Intrinsic::nvvm_barrier0_or:
          case Intrinsic::nvvm_membar_cta:
          case Intrinsic::nvvm_membar_gl:
          case Intrinsic::nvvm_membar_sys:
          case Intrinsic::amdgcn_s_barrier:
          case Intrinsic::assume:
          case Intrinsic::stacksave:
          case Intrinsic::stackrestore:
          case Intrinsic::lifetime_start:
          case Intrinsic::lifetime_end:
          case Intrinsic::dbg_addr:
          case Intrinsic::dbg_declare:
          case Intrinsic::dbg_value:
          case Intrinsic::invariant_start:
          case Intrinsic::invariant_end:
          case Intrinsic::var_annotation:
          case Intrinsic::ptr_annotation:
          case Intrinsic::annotation:
          case Intrinsic::codeview_annotation:
          case Intrinsic::expect:
          case Intrinsic::type_test:
          case Intrinsic::donothing:
          case Intrinsic::prefetch:
          case Intrinsic::trap:
#if LLVM_VERSION_MAJOR >= 8
          case Intrinsic::is_constant:
#endif
            noUse = true;
            break;
          default:
            break;
          }
          if (noUse)
            return false;
        }
      }

      Value *memval = Val;

      // BasicAA stupidy assumes that non-pointer's don't alias
      // if this is a nonpointer, use something else to force alias
      // consideration
      if (!memval->getType()->isPointerTy()) {
        if (auto ci = dyn_cast<CastInst>(Val)) {
          if (ci->getOperand(0)->getType()->isPointerTy()) {
            memval = ci->getOperand(0);
          }
        }
        for (auto user : Val->users()) {
          if (isa<CastInst>(user) && user->getType()->isPointerTy()) {
            memval = user;
            break;
          }
        }
      }

#if LLVM_VERSION_MAJOR >= 12
      auto AARes = AA.getModRefInfo(
          I, MemoryLocation(memval, LocationSize::beforeOrAfterPointer()));
#elif LLVM_VERSION_MAJOR >= 9
      auto AARes =
          AA.getModRefInfo(I, MemoryLocation(memval, LocationSize::unknown()));
#else
      auto AARes = AA.getModRefInfo(
          I, MemoryLocation(memval, MemoryLocation::UnknownSize));
#endif

      // Still having failed to replace the location used by AA, fall back to
      // getModref against any location.
      if (!memval->getType()->isPointerTy()) {
        if (auto CB = dyn_cast<CallInst>(I)) {
          AARes = createModRefInfo(AA.getModRefBehavior(CB));
        } else {
          bool mayRead = I->mayReadFromMemory();
          bool mayWrite = I->mayWriteToMemory();
          AARes = mayRead ? (mayWrite ? ModRefInfo::ModRef : ModRefInfo::Ref)
                          : (mayWrite ? ModRefInfo::Mod : ModRefInfo::NoModRef);
        }
      }

      // TODO this aliasing information is too conservative, the question
      // isn't merely aliasing but whether there is a path for THIS value to
      // eventually be loaded by it not simply because there isnt aliasing

      // If we haven't already shown a potentially active load
      // check if this loads the given value and is active
      if (!potentiallyActiveLoad && isRefSet(AARes)) {
        if (EnzymePrintActivity)
          llvm::errs() << "potential active load: " << *I << "\n";
        if (isa<LoadInst>(I) || (isa<IntrinsicInst>(I) &&
                                 (cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldu_global_i ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldu_global_p ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldu_global_f ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldg_global_i ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldg_global_p ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldg_global_f))) {
          // If the ref'ing value is a load check if the loaded value is
          // active
          if (!Hypothesis->isConstantValue(TR, I)) {
            potentiallyActiveLoad = true;
            if (TR.query(I)[{-1}].isPossiblePointer()) {
              if (EnzymePrintActivity)
                llvm::errs()
                    << "potential active store via pointer in load: " << *I
                    << " of " << *Val << "\n";
              potentiallyActiveStore = true;
            }
          }
        } else if (auto MTI = dyn_cast<MemTransferInst>(I)) {
          if (!Hypothesis->isConstantValue(TR, MTI->getArgOperand(0))) {
            potentiallyActiveLoad = true;
            if (TR.query(Val)[{-1, -1}].isPossiblePointer()) {
              if (EnzymePrintActivity)
                llvm::errs()
                    << "potential active store via pointer in memcpy: " << *I
                    << " of " << *Val << "\n";
              potentiallyActiveStore = true;
            }
          }
        } else {
          // Otherwise fallback and check any part of the instruction is
          // active
          // TODO: note that this can be optimized (especially for function
          // calls)
          // Notably need both to check the result and instruction since
          // A load that has as result an active pointer is not an active
          // instruction, but does have an active value
          if (!Hypothesis->isConstantInstruction(TR, I) ||
              (I != Val && !Hypothesis->isConstantValue(TR, I))) {
            potentiallyActiveLoad = true;
            // If this a potential pointer of pointer AND
            //     double** Val;
            //
            if (TR.query(Val)[{-1, -1}].isPossiblePointer()) {
              // If this instruction either:
              //   1) can actively store into the inner pointer, even
              //      if it doesn't store into the outer pointer. Actively
              //      storing into the outer pointer is handled by the isMod
              //      case.
              //        I(double** readonly Val, double activeX) {
              //            double* V0 = Val[0]
              //            V0 = activeX;
              //        }
              //   2) may return an active pointer loaded from Val
              //        double* I = *Val;
              //        I[0] = active;
              //
              if ((I->mayWriteToMemory() &&
                   !Hypothesis->isConstantInstruction(TR, I)) ||
                  (!Hypothesis->DeducingPointers.count(I) &&
                   !Hypothesis->isConstantValue(TR, I) &&
                   TR.query(I)[{-1}].isPossiblePointer())) {
                if (EnzymePrintActivity)
                  llvm::errs() << "potential active store via pointer in "
                                  "unknown inst: "
                               << *I << " of " << *Val << "\n";
                potentiallyActiveStore = true;
              }
            }
          }
        }
      }
      if ((!potentiallyActiveStore || !potentialStore) && isModSet(AARes)) {
        if (EnzymePrintActivity)
          llvm::errs() << "potential active store: " << *I << " Val=" << *Val
                       << "\n";
        if (auto SI = dyn_cast<StoreInst>(I)) {
          bool cop = !Hypothesis->isConstantValue(TR, SI->getValueOperand());
          if (EnzymePrintActivity)
            llvm::errs() << " -- store potential activity: " << (int)cop
                         << " - " << *SI << " of "
                         << " Val=" << *Val << "\n";
          potentialStore = true;
          if (cop)
            potentiallyActiveStore = true;
        } else if (auto MTI = dyn_cast<MemTransferInst>(I)) {
          bool cop = !Hypothesis->isConstantValue(TR, MTI->getArgOperand(1));
          potentialStore = true;
          if (cop)
            potentiallyActiveStore = true;
        } else {
          // Otherwise fallback and check if the instruction is active
          // TODO: note that this can be optimized (especially for function
          // calls)
          auto cop = !Hypothesis->isConstantInstruction(TR, I);
          if (EnzymePrintActivity)
            llvm::errs() << " -- unknown store potential activity: " << (int)cop
                         << " - " << *I << " of "
                         << " Val=" << *Val << "\n";
          potentialStore = true;
          if (cop)
            potentiallyActiveStore = true;
        }
      }
      if (potentiallyActiveStore && potentiallyActiveLoad)
        return true;
      return false;
    };

    // Search through all the instructions in this function
    // for potential loads / stores of this value.
    //
    // We can choose to only look at potential follower instructions
    // if the value is created by the instruction (alloca, noalias)
    // since no potentially active store to the same location can occur
    // prior to its creation. Otherwise, check all instructions in the
    // function as a store to an aliasing location may have occured
    // prior to the instruction generating the value.

    if (auto VI = dyn_cast<AllocaInst>(Val)) {
      allFollowersOf(VI, checkActivity);
    } else if (auto VI = dyn_cast<CallInst>(Val)) {
      if (VI->hasRetAttr(Attribute::NoAlias))
        allFollowersOf(VI, checkActivity);
      else {
        for (BasicBlock &BB : *TR.getFunction()) {
          if (notForAnalysis.count(&BB))
            continue;
          for (Instruction &I : BB) {
            if (checkActivity(&I))
              goto activeLoadAndStore;
          }
        }
      }
    } else if (isa<Argument>(Val) || isa<Instruction>(Val)) {
      for (BasicBlock &BB : *TR.getFunction()) {
        if (notForAnalysis.count(&BB))
          continue;
        for (Instruction &I : BB) {
          if (checkActivity(&I))
            goto activeLoadAndStore;
        }
      }
    } else {
      llvm::errs() << "unknown pointer value type: " << *Val << "\n";
      assert(0 && "unknown pointer value type");
      llvm_unreachable("unknown pointer value type");
    }

  activeLoadAndStore:;
    if (EnzymePrintActivity)
      llvm::errs() << " </MEMSEARCH" << (int)directions << ">" << *Val
                   << " potentiallyActiveLoad=" << potentiallyActiveLoad
                   << " potentiallyActiveStore=" << potentiallyActiveStore
                   << " potentialStore=" << potentialStore << "\n";
    if (potentiallyActiveLoad && potentiallyActiveStore) {
      insertAllFrom(TR, *Hypothesis, Val);
      // TODO have insertall dependence on this
      if (TmpOrig != Val)
        ReEvaluateValueIfInactiveValue[TmpOrig].insert(Val);
      return false;
    } else {
      // We now know that there isn't a matching active load/store pair in this
      // function. Now the only way that this memory can facilitate a transfer
      // of active information is if it is done outside of the function

      // This can happen if either:
      // a) the memory had an active load or store before this function was
      // called b) the memory had an active load or store after this function
      // was called

      // Case a) can occur if:
      //    1) this memory came from an active global
      //    2) this memory came from an active argument
      //    3) this memory came from a load from active memory
      // In other words, assuming this value is inactive, going up this
      // location's argument must be inactive

      assert(UpHypothesis);
      // UpHypothesis.ConstantValues.insert(val);
      if (DeducingPointers.size() == 0)
        UpHypothesis->insertConstantsFrom(TR, *Hypothesis);
      for (auto V : DeducingPointers) {
        UpHypothesis->InsertConstantValue(TR, V);
      }
      assert(directions & UP);
      bool ActiveUp = !isa<Argument>(Val) &&
                      !UpHypothesis->isInstructionInactiveFromOrigin(TR, Val);

      // Case b) can occur if:
      //    1) this memory is used as part of an active return
      //    2) this memory is stored somewhere

      // We never verify that an origin wasn't stored somewhere or returned.
      // to remedy correctness for now let's do something extremely simple
      std::shared_ptr<ActivityAnalyzer> DownHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, DOWN));
      DownHypothesis->ConstantValues.insert(Val);
      DownHypothesis->insertConstantsFrom(TR, *Hypothesis);
      bool ActiveDown =
          DownHypothesis->isValueActivelyStoredOrReturned(TR, Val);
      // BEGIN TEMPORARY

      if (!ActiveDown && TmpOrig != Val) {

        if (isa<Argument>(TmpOrig) || isa<GlobalVariable>(TmpOrig) ||
            isa<AllocaInst>(TmpOrig) ||
            (isCalledFunction(TmpOrig) &&
             isAllocationFunction(*isCalledFunction(TmpOrig), TLI))) {
          std::shared_ptr<ActivityAnalyzer> DownHypothesis2 =
              std::shared_ptr<ActivityAnalyzer>(
                  new ActivityAnalyzer(*DownHypothesis, DOWN));
          DownHypothesis2->ConstantValues.insert(TmpOrig);
          if (DownHypothesis2->isValueActivelyStoredOrReturned(TR, TmpOrig)) {
            if (EnzymePrintActivity)
              llvm::errs() << " active from ivasor: " << *TmpOrig << "\n";
            ActiveDown = true;
          }
        } else {
          // unknown origin that could've been stored/returned/etc
          if (EnzymePrintActivity)
            llvm::errs() << " active from unknown origin: " << *TmpOrig << "\n";
          ActiveDown = true;
        }
      }

      // END TEMPORARY

      // We can now consider the three places derivative information can be
      // transferred
      //   Case A) From the origin
      //   Case B) Though the return
      //   Case C) Within the function (via either load or store)

      bool ActiveMemory = false;

      // If it is transferred via active origin and return, clearly this is
      // active
      ActiveMemory |= (ActiveUp && ActiveDown);

      // If we come from an active origin and load, memory is clearly active
      ActiveMemory |= (ActiveUp && potentiallyActiveLoad);

      // If we come from an active origin and only store into it, it changes
      // future state
      ActiveMemory |= (ActiveUp && potentialStore);

      // If we go to an active return and store active memory, this is active
      ActiveMemory |= (ActiveDown && potentialStore);
      // Actually more generally, if we are ActiveDown (returning memory that is
      // used) in active return, we must be active. This is necessary to ensure
      // mallocs have their differential shadows created when returned [TODO
      // investigate more]
      ActiveMemory |= ActiveDown;

      // If we go to an active return and only load it, however, that doesnt
      // transfer derivatives and we can say this memory is inactive

      if (EnzymePrintActivity)
        llvm::errs() << " @@MEMSEARCH" << (int)directions << ">" << *Val
                     << " potentiallyActiveLoad=" << potentiallyActiveLoad
                     << " potentialStore=" << potentialStore
                     << " ActiveUp=" << ActiveUp << " ActiveDown=" << ActiveDown
                     << " ActiveMemory=" << ActiveMemory << "\n";

      if (ActiveMemory) {
        ActiveValues.insert(Val);
        assert(Hypothesis->directions == directions);
        assert(Hypothesis->ActiveValues.count(Val));
        insertAllFrom(TR, *Hypothesis, Val);
        if (TmpOrig != Val)
          ReEvaluateValueIfInactiveValue[TmpOrig].insert(Val);
        return false;
      } else {
        InsertConstantValue(TR, Val);
        insertConstantsFrom(TR, *Hypothesis);
        if (DeducingPointers.size() == 0)
          insertConstantsFrom(TR, *UpHypothesis);
        insertConstantsFrom(TR, *DownHypothesis);
        return true;
      }
    }
  }

  // For all non-pointers, it is now sufficient to simply prove that
  // either activity does not flow in, or activity does not flow out
  // This alone cuts off the flow (being unable to flow through memory)

  // Not looking at uses to prove inactive (definition of up), if the creator of
  // this value is inactive, we are inactive Since we won't look at uses to
  // prove, we can inductively assume this is inactive
  if (directions & UP) {
    if (directions == UP && !isa<PHINode>(Val)) {
      if (isInstructionInactiveFromOrigin(TR, Val)) {
        InsertConstantValue(TR, Val);
        return true;
      } else if (auto I = dyn_cast<Instruction>(Val)) {
        if (directions == 3) {
          for (auto &op : I->operands()) {
            if (!UpHypothesis->isConstantValue(TR, op)) {
              ReEvaluateValueIfInactiveValue[op].insert(I);
            }
          }
        }
      }
    } else {
      UpHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
      UpHypothesis->ConstantValues.insert(Val);
      if (UpHypothesis->isInstructionInactiveFromOrigin(TR, Val)) {
        insertConstantsFrom(TR, *UpHypothesis);
        InsertConstantValue(TR, Val);
        return true;
      } else if (auto I = dyn_cast<Instruction>(Val)) {
        if (directions == 3) {
          for (auto &op : I->operands()) {
            if (!UpHypothesis->isConstantValue(TR, op)) {
              ReEvaluateValueIfInactiveValue[op].insert(I);
            }
          }
        }
      }
    }
  }

  if (directions & DOWN) {
    // Not looking at users to prove inactive (definition of down)
    // If all users are inactive, this is therefore inactive.
    // Since we won't look at origins to prove, we can inductively assume this
    // is inactive

    // As an optimization if we are going down already
    // and we won't use ourselves (done by PHI's), we
    // dont need to inductively assume we're true
    // and can instead use this object!
    if (directions == DOWN && !isa<PHINode>(Val)) {
      if (isValueInactiveFromUsers(TR, Val, UseActivity::None)) {
        if (UpHypothesis)
          insertConstantsFrom(TR, *UpHypothesis);
        InsertConstantValue(TR, Val);
        return true;
      }
    } else {
      auto DownHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, DOWN));
      DownHypothesis->ConstantValues.insert(Val);
      if (DownHypothesis->isValueInactiveFromUsers(TR, Val,
                                                   UseActivity::None)) {
        insertConstantsFrom(TR, *DownHypothesis);
        if (UpHypothesis)
          insertConstantsFrom(TR, *UpHypothesis);
        InsertConstantValue(TR, Val);
        return true;
      }
    }
  }

  if (EnzymePrintActivity)
    llvm::errs() << " Value nonconstant (couldn't disprove)[" << (int)directions
                 << "]" << *Val << "\n";
  ActiveValues.insert(Val);
  return false;
}

/// Is the instruction guaranteed to be inactive because of its operands
bool ActivityAnalyzer::isInstructionInactiveFromOrigin(TypeResults const &TR,
                                                       llvm::Value *val) {
  // Must be an analyzer only searching up
  assert(directions == UP);
  assert(!isa<Argument>(val));
  assert(!isa<GlobalVariable>(val));

  // Not an instruction and thus not legal to search for activity via operands
  if (!isa<Instruction>(val)) {
    llvm::errs() << "unknown pointer source: " << *val << "\n";
    assert(0 && "unknown pointer source");
    llvm_unreachable("unknown pointer source");
    return false;
  }

  Instruction *inst = cast<Instruction>(val);
  if (EnzymePrintActivity)
    llvm::errs() << " < UPSEARCH" << (int)directions << ">" << *inst << "\n";

  // cpuid is explicitly an inactive instruction
  if (auto call = dyn_cast<CallInst>(inst)) {
#if LLVM_VERSION_MAJOR >= 11
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand())) {
#else
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledValue())) {
#endif
      if (StringRef(iasm->getAsmString()).contains("cpuid")) {
        if (EnzymePrintActivity)
          llvm::errs() << " constant instruction from known cpuid instruction "
                       << *inst << "\n";
        return true;
      }
    }
  }

  if (isa<MemSetInst>(inst)) {
    // memset's are definitionally inactive since
    // they copy a byte which cannot be active
    if (EnzymePrintActivity)
      llvm::errs() << " constant instruction as memset " << *inst << "\n";
    return true;
  }

  if (auto SI = dyn_cast<StoreInst>(inst)) {
    // if either src or dst is inactive, there cannot be a transfer of active
    // values and thus the store is inactive
    if (isConstantValue(TR, SI->getValueOperand()) ||
        isConstantValue(TR, SI->getPointerOperand())) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction as store operand is inactive "
                     << *inst << "\n";
      return true;
    }
  }

  if (auto MTI = dyn_cast<MemTransferInst>(inst)) {
    // if either src or dst is inactive, there cannot be a transfer of active
    // values and thus the store is inactive
    if (isConstantValue(TR, MTI->getArgOperand(0)) ||
        isConstantValue(TR, MTI->getArgOperand(1))) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction as memset " << *inst << "\n";
      return true;
    }
  }

  if (auto op = dyn_cast<CallInst>(inst)) {
    if (op->hasFnAttr("enzyme_inactive")) {
      return true;
    }
    // Calls to print/assert/cxa guard are definitionally inactive
    llvm::Value *callVal;
#if LLVM_VERSION_MAJOR >= 11
    callVal = op->getCalledOperand();
#else
    callVal = op->getCalledValue();
#endif
    if (Function *called = getFunctionFromCall(op)) {
      if (called->hasFnAttribute("enzyme_inactive")) {
        return true;
      }
      if (called->getName() == "free" || called->getName() == "_ZdlPv" ||
          called->getName() == "_ZdlPvm" || called->getName() == "munmap") {
        return true;
      }

#if LLVM_VERSION_MAJOR >= 9
      auto dName = demangle(called->getName().str());
      for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
        if (StringRef(dName).startswith(FuncName)) {
          return true;
        }
      }
#endif

      for (auto FuncName : KnownInactiveFunctionsStartingWith) {
        if (called->getName().startswith(FuncName)) {
          return true;
        }
      }

      for (auto FuncName : KnownInactiveFunctionsContains) {
        if (called->getName().contains(FuncName)) {
          return true;
        }
      }

      if (KnownInactiveFunctions.count(called->getName().str()) ||
          MPIInactiveCommAllocators.find(called->getName().str()) !=
              MPIInactiveCommAllocators.end()) {
        if (EnzymePrintActivity)
          llvm::errs() << "constant(" << (int)directions
                       << ") up-knowninactivecall " << *inst << "\n";
        return true;
      }

      if (called->getIntrinsicID() == Intrinsic::trap)
        return true;

      // If requesting empty unknown functions to be considered inactive, abide
      // by those rules
      if (EnzymeEmptyFnInactive && called->empty() &&
          !hasMetadata(called, "enzyme_gradient") &&
          !hasMetadata(called, "enzyme_derivative") &&
          !isAllocationFunction(*called, TLI) &&
          !isDeallocationFunction(*called, TLI) && !isa<IntrinsicInst>(op)) {
        if (EnzymePrintActivity)
          llvm::errs() << "constant(" << (int)directions << ") up-emptyconst "
                       << *inst << "\n";
        return true;
      }
    } else if (!isa<Constant>(callVal) && isConstantValue(TR, callVal)) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ") up-constfn "
                     << *inst << " - " << *callVal << "\n";
      return true;
    }
  }
  // Intrinsics known always to be inactive
  if (auto II = dyn_cast<IntrinsicInst>(inst)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::nvvm_barrier0:
    case Intrinsic::nvvm_barrier0_popc:
    case Intrinsic::nvvm_barrier0_and:
    case Intrinsic::nvvm_barrier0_or:
    case Intrinsic::nvvm_membar_cta:
    case Intrinsic::nvvm_membar_gl:
    case Intrinsic::nvvm_membar_sys:
    case Intrinsic::amdgcn_s_barrier:
    case Intrinsic::assume:
    case Intrinsic::stacksave:
    case Intrinsic::stackrestore:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::dbg_addr:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::var_annotation:
    case Intrinsic::ptr_annotation:
    case Intrinsic::annotation:
    case Intrinsic::codeview_annotation:
    case Intrinsic::expect:
    case Intrinsic::type_test:
    case Intrinsic::donothing:
    case Intrinsic::prefetch:
#if LLVM_VERSION_MAJOR >= 8
    case Intrinsic::is_constant:
#endif
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ") up-intrinsic "
                     << *inst << "\n";
      return true;
    default:
      break;
    }
  }

  if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
    // A gep's only args that could make it active is the pointer operand
    if (isConstantValue(TR, gep->getPointerOperand())) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ") up-gep " << *inst
                     << "\n";
      return true;
    }
    return false;
  } else if (auto ci = dyn_cast<CallInst>(inst)) {
    bool seenuse = false;

    propagateArgumentInformation(TLI, *ci, [&](Value *a) {
      if (!isConstantValue(TR, a)) {
        seenuse = true;
        if (EnzymePrintActivity)
          llvm::errs() << "nonconstant(" << (int)directions << ")  up-call "
                       << *inst << " op " << *a << "\n";
        return true;
      }
      return false;
    });
    if (EnzymeGlobalActivity) {
      if (!ci->onlyAccessesArgMemory() && !ci->doesNotAccessMemory()) {

        Function *called = getFunctionFromCall(ci);
        if (!called || (!isCertainPrintMallocOrFree(called) &&
                        !isMemFreeLibMFunction(called->getName()))) {
          if (EnzymePrintActivity)
            llvm::errs() << "nonconstant(" << (int)directions << ")  up-global "
                         << *inst << "\n";
          seenuse = true;
        }
      }
    }

    if (!seenuse) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ")  up-call:" << *inst
                     << "\n";
      return true;
    }
    return !seenuse;
  } else if (auto si = dyn_cast<SelectInst>(inst)) {

    if (isConstantValue(TR, si->getTrueValue()) &&
        isConstantValue(TR, si->getFalseValue())) {

      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ") up-sel:" << *inst
                     << "\n";
      return true;
    }
    return false;
  } else if (isa<SIToFPInst>(inst) || isa<UIToFPInst>(inst) ||
             isa<FPToSIInst>(inst) || isa<FPToUIInst>(inst)) {

    if (EnzymePrintActivity)
      llvm::errs() << "constant(" << (int)directions << ") up-fpcst:" << *inst
                   << "\n";
    return true;
  } else {
    bool seenuse = false;
    //! TODO does not consider reading from global memory that is active and not
    //! an argument
    for (auto &a : inst->operands()) {
      bool hypval = isConstantValue(TR, a);
      if (!hypval) {
        if (EnzymePrintActivity)
          llvm::errs() << "nonconstant(" << (int)directions << ")  up-inst "
                       << *inst << " op " << *a << "\n";
        seenuse = true;
        break;
      }
    }

    if (!seenuse) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ")  up-inst:" << *inst
                     << "\n";
      return true;
    }
    return false;
  }
}

/// Is the value free of any active uses
bool ActivityAnalyzer::isValueInactiveFromUsers(TypeResults const &TR,
                                                llvm::Value *val,
                                                UseActivity PUA,
                                                Instruction **FoundInst) {
  assert(directions & DOWN);
  // Must be an analyzer only searching down, unless used outside
  // assert(directions == DOWN);

  // To ensure we can call down

  if (EnzymePrintActivity)
    llvm::errs() << " <Value USESEARCH" << (int)directions << ">" << *val
                 << " UA=" << (int)PUA << "\n";

  bool seenuse = false;
  // user, predecessor
  std::deque<std::tuple<User *, Value *, UseActivity>> todo;
  for (const auto a : val->users()) {
    todo.push_back(std::make_tuple(a, val, PUA));
  }
  std::set<std::tuple<User *, Value *, UseActivity>> done = {};

  while (todo.size()) {
    auto pair = todo.front();
    todo.pop_front();
    if (done.count(pair))
      continue;
    done.insert(pair);
    User *a = std::get<0>(pair);
    Value *parent = std::get<1>(pair);
    UseActivity UA = std::get<2>(pair);

    if (UA == UseActivity::OnlyStores && isa<LoadInst>(a))
      continue;

    // Only ignore stores to the operand, not storing the operand
    // somewhere
    if (auto SI = dyn_cast<StoreInst>(a)) {
      if (UA == UseActivity::OnlyLoads) {
        if (SI->getValueOperand() != parent) {
          continue;
        }
      }
      if (PUA == UseActivity::OnlyLoads) {
        auto TmpOrig =
#if LLVM_VERSION_MAJOR >= 12
            getUnderlyingObject(SI->getPointerOperand(), 100);
#else
            GetUnderlyingObject(SI->getPointerOperand(),
                                TR.getFunction()->getParent()->getDataLayout(),
                                100);
#endif
        if (TmpOrig == val) {
          continue;
        }
      }
    }

    if (EnzymePrintActivity)
      llvm::errs() << "      considering use of " << *val << " - " << *a
                   << "\n";

    if (!isa<Instruction>(a)) {
      if (auto CE = dyn_cast<ConstantExpr>(a)) {
        for (auto u : CE->users()) {
          todo.push_back(std::make_tuple(u, (Value *)CE, UA));
        }
        continue;
      }
      if (isa<ConstantData>(a)) {
        continue;
      }

      if (EnzymePrintActivity)
        llvm::errs() << "      unknown non instruction use of " << *val << " - "
                     << *a << "\n";
      return false;
    }

    if (isa<AllocaInst>(a)) {
      if (EnzymePrintActivity)
        llvm::errs() << "found constant(" << (int)directions
                     << ")  allocainst use:" << *val << " user " << *a << "\n";
      continue;
    }

    if (isa<SIToFPInst>(a) || isa<UIToFPInst>(a) || isa<FPToSIInst>(a) ||
        isa<FPToUIInst>(a)) {
      if (EnzymePrintActivity)
        llvm::errs() << "found constant(" << (int)directions
                     << ")  si-fp use:" << *val << " user " << *a << "\n";
      continue;
    }

    // if this instruction is in a different function, conservatively assume
    // it is active
    Function *InstF = cast<Instruction>(a)->getParent()->getParent();
    while (PPC.CloneOrigin.find(InstF) != PPC.CloneOrigin.end())
      InstF = PPC.CloneOrigin[InstF];

    Function *F = TR.getFunction();
    while (PPC.CloneOrigin.find(F) != PPC.CloneOrigin.end())
      F = PPC.CloneOrigin[F];

    if (InstF != F) {
      if (EnzymePrintActivity)
        llvm::errs() << "found use in different function(" << (int)directions
                     << ")  val:" << *val << " user " << *a << " in "
                     << InstF->getName() << "@" << InstF
                     << " self: " << F->getName() << "@" << F << "\n";
      return false;
    }
    if (cast<Instruction>(a)->getParent()->getParent() != TR.getFunction())
      continue;

    // This use is only active if specified
    if (isa<ReturnInst>(a)) {
      if (ActiveReturns == DIFFE_TYPE::CONSTANT) {
        continue;
      } else {
        return false;
      }
    }

    if (auto call = dyn_cast<CallInst>(a)) {
      bool ConstantArg = isFunctionArgumentConstant(call, parent);
      if (ConstantArg) {
        if (EnzymePrintActivity) {
          llvm::errs() << "Value found constant callinst use:" << *val
                       << " user " << *call << "\n";
        }
        continue;
      }
    }

    // If this doesn't write to memory this can only be an active use
    // if its return is used in an active way, therefore add this to
    // the list of users to analyze
    if (auto I = dyn_cast<Instruction>(a)) {
      if (notForAnalysis.count(I->getParent())) {
        if (EnzymePrintActivity) {
          llvm::errs() << "Value found constant unreachable inst use:" << *val
                       << " user " << *I << "\n";
        }
        continue;
      }
      if (ConstantInstructions.count(I) &&
          (I->getType()->isVoidTy() || I->getType()->isTokenTy() ||
           ConstantValues.count(I))) {
        if (EnzymePrintActivity) {
          llvm::errs() << "Value found constant inst use:" << *val << " user "
                       << *I << "\n";
        }
        continue;
      }
      if (!I->mayWriteToMemory()) {
        if (TR.intType(1, I, /*errIfNotFound*/ false).isIntegral()) {
          continue;
        }
        UseActivity NU = UA;
        if (UA == UseActivity::OnlyLoads || UA == UseActivity::OnlyStores) {
          if (!isa<PHINode>(I) && !isa<CastInst>(I) &&
              !isa<GetElementPtrInst>(I) && !isa<BinaryOperator>(I))
            NU = UseActivity::None;
        }

        for (auto u : I->users()) {
          todo.push_back(std::make_tuple(u, (Value *)I, NU));
        }
        continue;
      }

      if (FoundInst)
        *FoundInst = I;
    }

    if (EnzymePrintActivity)
      llvm::errs() << "Value nonconstant inst (uses):" << *val << " user " << *a
                   << "\n";
    seenuse = true;
    break;
  }

  if (EnzymePrintActivity)
    llvm::errs() << " </Value USESEARCH" << (int)directions
                 << " const=" << (!seenuse) << ">" << *val << "\n";
  return !seenuse;
}

/// Is the value potentially actively returned or stored
bool ActivityAnalyzer::isValueActivelyStoredOrReturned(TypeResults const &TR,
                                                       llvm::Value *val,
                                                       bool outside) {
  // Must be an analyzer only searching down
  if (!outside)
    assert(directions == DOWN);

  bool ignoreStoresInto = true;
  auto key = std::make_pair(ignoreStoresInto, val);
  if (StoredOrReturnedCache.find(key) != StoredOrReturnedCache.end()) {
    return StoredOrReturnedCache[key];
  }

  if (EnzymePrintActivity)
    llvm::errs() << " <ASOR" << (int)directions
                 << " ignoreStoresinto=" << ignoreStoresInto << ">" << *val
                 << "\n";

  StoredOrReturnedCache[key] = false;

  for (const auto a : val->users()) {
    if (isa<AllocaInst>(a)) {
      continue;
    }
    // Loading a value prevents its pointer from being captured
    if (isa<LoadInst>(a)) {
      continue;
    }

    if (isa<ReturnInst>(a)) {
      if (ActiveReturns == DIFFE_TYPE::CONSTANT)
        continue;

      if (EnzymePrintActivity)
        llvm::errs() << " </ASOR" << (int)directions
                     << " ignoreStoresInto=" << ignoreStoresInto << ">"
                     << " active from-ret>" << *val << "\n";
      StoredOrReturnedCache[key] = true;
      return true;
    }

    if (auto call = dyn_cast<CallInst>(a)) {
      if (!couldFunctionArgumentCapture(call, val)) {
        continue;
      }
      bool ConstantArg = isFunctionArgumentConstant(call, val);
      if (ConstantArg) {
        continue;
      }
    }

    if (auto SI = dyn_cast<StoreInst>(a)) {
      // If we are being stored into, not storing this value
      // this case can be skipped
      if (SI->getValueOperand() != val) {
        if (!ignoreStoresInto) {
          // Storing into active value, return true
          if (!isConstantValue(TR, SI->getValueOperand())) {
            StoredOrReturnedCache[key] = true;
            if (EnzymePrintActivity)
              llvm::errs() << " </ASOR" << (int)directions
                           << " ignoreStoresInto=" << ignoreStoresInto
                           << " active from-store>" << *val
                           << " store into=" << *SI << "\n";
            return true;
          }
        }
        continue;
      } else {
        // Storing into active memory, return true
        if (!isConstantValue(TR, SI->getPointerOperand())) {
          StoredOrReturnedCache[key] = true;
          if (EnzymePrintActivity)
            llvm::errs() << " </ASOR" << (int)directions
                         << " ignoreStoresInto=" << ignoreStoresInto
                         << " active from-store>" << *val << " store=" << *SI
                         << "\n";
          return true;
        }
        continue;
      }
    }

    if (auto inst = dyn_cast<Instruction>(a)) {
      if (!inst->mayWriteToMemory() ||
          (isa<CallInst>(inst) && AA.onlyReadsMemory(cast<CallInst>(inst)))) {
        // if not written to memory and returning a known constant, this
        // cannot be actively returned/stored
        if (inst->getParent()->getParent() == TR.getFunction() &&
            isConstantValue(TR, a)) {
          continue;
        }
        // if not written to memory and returning a value itself
        // not actively stored or returned, this is not actively
        // stored or returned
        if (!isValueActivelyStoredOrReturned(TR, a, outside)) {
          continue;
        }
      }
    }

    if (auto F = isCalledFunction(a)) {
      if (isAllocationFunction(*F, TLI)) {
        // if not written to memory and returning a known constant, this
        // cannot be actively returned/stored
        if (isConstantValue(TR, a)) {
          continue;
        }
        // if not written to memory and returning a value itself
        // not actively stored or returned, this is not actively
        // stored or returned
        if (!isValueActivelyStoredOrReturned(TR, a, outside)) {
          continue;
        }
      } else if (isDeallocationFunction(*F, TLI)) {
        // freeing memory never counts
        continue;
      }
    }
    // fallback and conservatively assume that if the value is written to
    // it is written to active memory
    // TODO handle more memory instructions above to be less conservative

    if (EnzymePrintActivity)
      llvm::errs() << " </ASOR" << (int)directions
                   << " ignoreStoresInto=" << ignoreStoresInto
                   << " active from-unknown>" << *val << " - use=" << *a
                   << "\n";
    return StoredOrReturnedCache[key] = true;
  }

  if (EnzymePrintActivity)
    llvm::errs() << " </ASOR" << (int)directions
                 << " ignoreStoresInto=" << ignoreStoresInto << " inactive>"
                 << *val << "\n";
  return false;
}

void ActivityAnalyzer::InsertConstantInstruction(TypeResults const &TR,
                                                 llvm::Instruction *I) {
  ConstantInstructions.insert(I);
  auto found = ReEvaluateValueIfInactiveInst.find(I);
  if (found == ReEvaluateValueIfInactiveInst.end())
    return;
  auto set = std::move(ReEvaluateValueIfInactiveInst[I]);
  ReEvaluateValueIfInactiveInst.erase(I);
  for (auto toeval : set) {
    if (!ActiveValues.count(toeval))
      continue;
    ActiveValues.erase(toeval);
    if (EnzymePrintActivity)
      llvm::errs() << " re-evaluating activity of val " << *toeval
                   << " due to inst " << *I << "\n";
    isConstantValue(TR, toeval);
  }
}

void ActivityAnalyzer::InsertConstantValue(TypeResults const &TR,
                                           llvm::Value *V) {
  ConstantValues.insert(V);
  auto found = ReEvaluateValueIfInactiveValue.find(V);
  if (found != ReEvaluateValueIfInactiveValue.end()) {
    auto set = std::move(ReEvaluateValueIfInactiveValue[V]);
    ReEvaluateValueIfInactiveValue.erase(V);
    for (auto toeval : set) {
      if (!ActiveValues.count(toeval))
        continue;
      ActiveValues.erase(toeval);
      if (EnzymePrintActivity)
        llvm::errs() << " re-evaluating activity of val " << *toeval
                     << " due to value " << *V << "\n";
      isConstantValue(TR, toeval);
    }
  }
  auto found2 = ReEvaluateInstIfInactiveValue.find(V);
  if (found2 != ReEvaluateInstIfInactiveValue.end()) {
    auto set = std::move(ReEvaluateInstIfInactiveValue[V]);
    ReEvaluateInstIfInactiveValue.erase(V);
    for (auto toeval : set) {
      if (!ActiveInstructions.count(toeval))
        continue;
      ActiveInstructions.erase(toeval);
      if (EnzymePrintActivity)
        llvm::errs() << " re-evaluating activity of inst " << *toeval
                     << " due to value " << *V << "\n";
      isConstantInstruction(TR, toeval);
    }
  }
}
