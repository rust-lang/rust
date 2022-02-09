//===- EnzymeClang.cpp - Automatic Differentiation Transformation Pass ----===//
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
// This file contains a clang plugin for Enzyme.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "../Enzyme.h"
#include "../PreserveNVVM.h"

#include "llvm/LinkAllPasses.h"

using namespace llvm;

// This function is of type PassManagerBuilder::ExtensionFn
static void loadPass(const PassManagerBuilder &Builder,
                     legacy::PassManagerBase &PM) {
  PM.add(createPreserveNVVMPass(/*Begin=*/true));
  PM.add(createGVNPass());
  PM.add(createSROAPass());
  PM.add(createEnzymePass(/*PostOpt*/ true));
  PM.add(createPreserveNVVMPass(/*Begin=*/false));
  PM.add(createGVNPass());
  PM.add(createSROAPass());
  PM.add(createLoopDeletionPass());
  PM.add(createGlobalOptimizerPass());
  // PM.add(SimplifyCFGPass());
}

static void loadNVVMPass(const PassManagerBuilder &Builder,
                         legacy::PassManagerBase &PM) {
  PM.add(createPreserveNVVMPass(/*Begin=*/true));
}

// These constructors add our pass to a list of global extensions.
static RegisterStandardPasses
    clangtoolLoader_Ox(PassManagerBuilder::EP_VectorizerStart, loadPass);
static RegisterStandardPasses
    clangtoolLoader_O0(PassManagerBuilder::EP_EnabledOnOptLevel0, loadPass);
static RegisterStandardPasses
    clangtoolLoader_OEarly(PassManagerBuilder::EP_EarlyAsPossible,
                           loadNVVMPass);

#if LLVM_VERSION_MAJOR >= 9

static void loadLTOPass(const PassManagerBuilder &Builder,
                        legacy::PassManagerBase &PM) {
  loadPass(Builder, PM);
  PassManagerBuilder Builder2 = Builder;
  Builder2.Inliner = nullptr;
  Builder2.LibraryInfo = nullptr;
  Builder2.ExportSummary = nullptr;
  Builder2.ImportSummary = nullptr;
  /*
  Builder2.LoopVectorize = false;
  Builder2.SLPVectorize = false;
  Builder2.DisableUnrollLoops = true;
  Builder2.RerollLoops = true;
  */
  const_cast<PassManagerBuilder &>(Builder2).populateModulePassManager(PM);
}

static RegisterStandardPasses
    clangtoolLoader_LTO(PassManagerBuilder::EP_FullLinkTimeOptimizationEarly,
                        loadLTOPass);

#include "clang/AST/Attr.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

template <typename ConsumerType>
class EnzymeAction : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) {
    return std::unique_ptr<clang::ASTConsumer>(new ConsumerType(CI));
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) {
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

class EnzymePlugin : public clang::ASTConsumer {
  clang::CompilerInstance &CI;

public:
  EnzymePlugin(clang::CompilerInstance &CI) : CI(CI) {}
  ~EnzymePlugin() {}
  bool HandleTopLevelDecl(clang::DeclGroupRef dg) override {
    using namespace clang;
    DeclGroupRef::iterator it;

    // Forcibly require emission of all libdevice
    for (it = dg.begin(); it != dg.end(); ++it) {
      auto FD = dyn_cast<FunctionDecl>(*it);
      if (!FD)
        continue;

      if (!FD->hasAttr<clang::CUDADeviceAttr>())
        continue;

      if (!FD->getIdentifier())
        continue;
      if (!StringRef(FD->getLocation().printToString(CI.getSourceManager()))
               .contains("/__clang_cuda_math.h"))
        continue;

      FD->addAttr(UsedAttr::CreateImplicit(CI.getASTContext()));
    }
    return true;
  }
};

// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<EnzymeAction<EnzymePlugin>>
    X("enzyme", "Enzyme Plugin");
#endif
