#ifndef TraceUtils_h
#define TraceUtils_h

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueMap.h"

#include "TraceInterface.h"
#include "Utils.h"

class TraceUtils {

private:
  TraceInterface *interface;
  llvm::Value *dynamic_interface = nullptr;
  llvm::Value *trace;
  llvm::Value *observations = nullptr;

public:
  ProbProgMode mode;
  llvm::Function *newFunc;
  llvm::Function *oldFunc;

public:
  llvm::ValueMap<const llvm::Value *, llvm::WeakTrackingVH> originalToNewFn;
  llvm::SmallPtrSetImpl<llvm::Function *> &generativeFunctions;

public:
  TraceUtils(ProbProgMode mode, bool has_dynamic_interface,
             llvm::Function *newFunc, llvm::Function *oldFunc,
             llvm::ValueMap<const llvm::Value *, llvm::WeakTrackingVH> vmap,
             llvm::SmallPtrSetImpl<llvm::Function *> &generativeFunctions);

  TraceUtils(ProbProgMode mode, bool has_dynamic_interface, llvm::Function *F,
             llvm::SmallPtrSetImpl<llvm::Function *> &generativeFunctions);

  ~TraceUtils();

public:
  TraceInterface *getTraceInterface();

  llvm::Value *getDynamicTraceInterface();

  bool hasDynamicTraceInterface();

  llvm::Value *getTrace();

  llvm::CallInst *CreateTrace(llvm::IRBuilder<> &Builder,
                              const llvm::Twine &Name = "trace");

  llvm::CallInst *InsertChoice(llvm::IRBuilder<> &Builder, llvm::Value *address,
                               llvm::Value *score, llvm::Value *choice);

  llvm::CallInst *InsertCall(llvm::IRBuilder<> &Builder, llvm::Value *address,
                             llvm::Value *subtrace);

  llvm::CallInst *GetTrace(llvm::IRBuilder<> &Builder, llvm::Value *address,
                           const llvm::Twine &Name = "");

  llvm::Instruction *GetChoice(llvm::IRBuilder<> &Builder, llvm::Value *address,
                               llvm::Type *choiceType,
                               const llvm::Twine &Name = "");

  llvm::Instruction *HasChoice(llvm::IRBuilder<> &Builder, llvm::Value *address,
                               const llvm::Twine &Name = "");

  llvm::Instruction *HasCall(llvm::IRBuilder<> &Builder, llvm::Value *address,
                             const llvm::Twine &Name = "");
};

#endif /* TraceUtils_h */
