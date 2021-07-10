#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

using namespace llvm;

cl::opt<std::string> BCPath("bcpath", cl::init(""), cl::Hidden,
                            cl::desc("Path to BC definitions"));

namespace {
class BCLoader : public ModulePass {
public:
  static char ID;
  BCLoader() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    std::set<std::string> bcfuncs = {"cblas_ddot"};
    for (std::string name : bcfuncs) {
      if (name == "cblas_ddot") {
        SMDiagnostic Err;
#if LLVM_VERSION_MAJOR <= 10
        auto BC = llvm::parseIRFile(
            BCPath + "/cblas_ddot_double.bc", Err, M.getContext(), true,
            M.getDataLayout().getStringRepresentation());
#else
        auto BC = llvm::parseIRFile(
            BCPath + "/cblas_ddot_double.bc", Err, M.getContext(),
            [&](StringRef) {
              return Optional<std::string>(
                  M.getDataLayout().getStringRepresentation());
            });
#endif
        if (!BC)
          Err.print("bcloader", llvm::errs());
        assert(BC);
        Linker L(M);
        L.linkInModule(std::move(BC));
      }
    }
    return true;
  }
};
} // namespace

char BCLoader::ID = 0;

static RegisterPass<BCLoader> X("bcloader",
                                "Link bitcode files for known functions");

ModulePass *createBCLoaderPass() { return new BCLoader(); }
