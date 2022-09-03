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

#include "clang/AST/Attr.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

template <typename ConsumerType>
class EnzymeAction final : public clang::PluginASTAction {
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

class EnzymePlugin final : public clang::ASTConsumer {
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
