#ifndef TraceGenerator_h
#define TraceGenerator_h

#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"

#include "EnzymeLogic.h"
#include "TraceUtils.h"

class TraceGenerator final : public llvm::InstVisitor<TraceGenerator> {
private:
  EnzymeLogic &Logic;
  TraceUtils *const tutils;
  ProbProgMode mode = tutils->mode;

public:
  TraceGenerator(EnzymeLogic &Logic, TraceUtils *const tutils);

  void visitCallInst(llvm::CallInst &call);
};

#endif /* TraceGenerator_h */
