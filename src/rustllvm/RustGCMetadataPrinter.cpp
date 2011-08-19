//===-- RustGCPrinter.cpp - Rust garbage collection map printer -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the emitter for the Rust garbage collection stack maps.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/GCs.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/GCMetadataPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include <cctype>
#include <map>

using namespace llvm;

namespace {
  enum RustGCMetaType {
    RGCMT_DestIndex,  // Type descriptor index -> type descriptor.
    RGCMT_SrcIndex,   // Value -> type descriptor index.
    RGCMT_Static      // Value with static type descriptor.
  };

  class RustGCMetadataPrinter : public GCMetadataPrinter {
  private:
    std::pair<RustGCMetaType,const Constant *>
      GetGCMetadataForRoot(const GCRoot &Root);
    void EmitGCMetadata(AsmPrinter &AP, MCStreamer &Out, GCRoot &Root);
    bool HandleDestIndex(const GCRoot &Root);
  public:
    void beginAssembly(AsmPrinter &AP) {};
    void finishAssembly(AsmPrinter &AP);
  };

  struct OrderedSymbol {
    unsigned Index;
    MCSymbol *Sym;

    OrderedSymbol(unsigned I, MCSymbol *S) : Index(I), Sym(S) {}

    static OrderedSymbol make(unsigned I, MCSymbol *S) {
      OrderedSymbol OS(I, S);
      return OS;
    }
  };
}

static GCMetadataPrinterRegistry::Add<RustGCMetadataPrinter>
X("rust", "Rust GC metadata printer");


typedef std::vector< std::pair< MCSymbol *,std::vector<GCRoot> > > RootMap;

std::pair<RustGCMetaType,const Constant *>
RustGCMetadataPrinter::GetGCMetadataForRoot(const GCRoot &Root) {
  const GlobalVariable *GCMetaVar =
    cast<const GlobalVariable>(Root.Metadata->stripPointerCasts());

  const Constant *GCMetaInit = GCMetaVar->getInitializer();
  if (isa<ConstantAggregateZero>(GCMetaInit)) {
    // "zeroinitializer": expand to (0, 0).
    IntegerType *I32 = IntegerType::get(GCMetaInit->getContext(), 32);
    ConstantInt *Zero = ConstantInt::get(I32, 0);
    return std::make_pair(RGCMT_DestIndex, Zero);
  }

  const ConstantStruct *GCMeta =
    cast<const ConstantStruct>(GCMetaVar->getInitializer());

  RustGCMetaType GCMetaType = (RustGCMetaType)
    (cast<const ConstantInt>(GCMeta->getOperand(0))->getZExtValue());
  const Constant *Payload = cast<const Constant>(GCMeta->getOperand(1));
  return std::make_pair(GCMetaType, Payload);
}

void RustGCMetadataPrinter::EmitGCMetadata(AsmPrinter &AP, MCStreamer &Out,
                                           GCRoot &Root) {
  int WordSize = AP.TM.getTargetData()->getPointerSize();

  std::pair<RustGCMetaType,const Constant *> Pair =
    GetGCMetadataForRoot(Root);
  const GlobalValue *Tydesc = 0;

  switch (Pair.first) {
  case RGCMT_DestIndex: // Dest index.
    assert(0 && "Dest index should not be here!");
  case RGCMT_SrcIndex:
    // TODO: Use the mapping to find the tydesc frame offset.
    Out.EmitIntValue(1, WordSize, 0);
    Out.EmitIntValue(0, WordSize, 0);
    return;
  case 2: // Static type descriptor.
    Out.EmitIntValue(0, WordSize, 0);
    Tydesc = cast<const GlobalValue>(Pair.second);
    break;
  }

  MCSymbol *TydescSym = AP.Mang->getSymbol(Tydesc);
  Out.EmitSymbolValue(TydescSym, WordSize, 0);
}

// Records the destination index of a type descriptor in the type descriptor
// map, if this GC root is a destination index. Returns true if the GC root is
// a destination index and false otherwise.
bool RustGCMetadataPrinter::HandleDestIndex(const GCRoot &Root) {
  std::pair<RustGCMetaType,const Constant *> Pair =
    GetGCMetadataForRoot(Root);
  return Pair.first == RGCMT_DestIndex; // TODO
}

void RustGCMetadataPrinter::finishAssembly(AsmPrinter &AP) {
  MCStreamer &Out = AP.OutStreamer;

  // Use the data section.
  Out.SwitchSection(AP.getObjFileLowering().getDataSection());

  // Iterate over each function.
  RootMap Map;

  iterator FI = begin(), FE = end();
  while (FI != FE) {
    GCFunctionInfo &GCFI = **FI;

    // Iterate over each safe point.
    GCFunctionInfo::iterator SPI = GCFI.begin(), SPE = GCFI.end();
    while (SPI != SPE) {
      std::vector<GCRoot> Roots;

      // Iterate over each live root.
      GCFunctionInfo::live_iterator LI = GCFI.live_begin(SPI);
      GCFunctionInfo::live_iterator LE = GCFI.live_end(SPI);
      while (LI != LE) {
        if (!HandleDestIndex(*LI))
          Roots.push_back(*LI);
        ++LI;
      }

      Map.push_back(std::make_pair(SPI->Label, Roots));
      ++SPI;
    }
    ++FI;
  }

  // Write out the map.
  Out.AddBlankLine();

  int WordSize = AP.TM.getTargetData()->getPointerSize();

  MCSymbol *SafePointSym = AP.GetExternalSymbolSymbol("rust_gc_safe_points");
  Out.EmitSymbolAttribute(SafePointSym, MCSA_Global);
  Out.EmitLabel(SafePointSym);
  Out.EmitIntValue(Map.size(), WordSize, 0);

  std::vector<MCSymbol *> FrameMapLabels;

  RootMap::iterator MI = Map.begin(), ME = Map.end();
  unsigned i = 0;
  while (MI != ME) {
    Out.EmitSymbolValue(MI->first, WordSize, 0);
    MCSymbol *FrameMapLabel = AP.GetTempSymbol("rust_frame_map_label", i);
    FrameMapLabels.push_back(FrameMapLabel);
    ++MI, ++i;
  }

  MI = Map.begin(), i = 0;
  while (MI != ME) {
    Out.EmitLabel(FrameMapLabels[i]);

    std::vector<GCRoot> &Roots = MI->second;
    Out.EmitIntValue(Roots.size(), WordSize, 0);

    std::vector<GCRoot>::iterator RI = Roots.begin(), RE = Roots.end();
    while (RI != RE) {
      Out.EmitIntValue(RI->StackOffset, WordSize, 0);
      EmitGCMetadata(AP, Out, *RI);
      ++RI;
    }

    ++MI, ++i;
  }
}

