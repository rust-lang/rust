//===- MachOObjectFile.cpp - Mach-O object file binding ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachOObjectFile class, which binds the MachOObject
// class to the generic ObjectFile wrapper.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cctype>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace object;

namespace llvm {

typedef MachOObject::LoadCommandInfo LoadCommandInfo;

class MachOObjectFile : public ObjectFile {
public:
  MachOObjectFile(MemoryBuffer *Object, MachOObject *MOO)
    : ObjectFile(Object),
      MachOObj(MOO),
      RegisteredStringTable(std::numeric_limits<uint32_t>::max()) {}

  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;
  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;

protected:
  virtual SymbolRef getSymbolNext(DataRefImpl Symb) const;
  virtual StringRef getSymbolName(DataRefImpl Symb) const;
  virtual uint64_t  getSymbolAddress(DataRefImpl Symb) const;
  virtual uint64_t  getSymbolSize(DataRefImpl Symb) const;
  virtual char      getSymbolNMTypeChar(DataRefImpl Symb) const;
  virtual bool      isSymbolInternal(DataRefImpl Symb) const;

  virtual SectionRef getSectionNext(DataRefImpl Sec) const;
  virtual StringRef  getSectionName(DataRefImpl Sec) const;
  virtual uint64_t   getSectionAddress(DataRefImpl Sec) const;
  virtual uint64_t   getSectionSize(DataRefImpl Sec) const;
  virtual StringRef  getSectionContents(DataRefImpl Sec) const;
  virtual bool       isSectionText(DataRefImpl Sec) const;

private:
  MachOObject *MachOObj;
  mutable uint32_t RegisteredStringTable;

  void moveToNextSection(DataRefImpl &DRI) const;
  void getSymbolTableEntry(DataRefImpl DRI,
                           InMemoryStruct<macho::SymbolTableEntry> &Res) const;
  void moveToNextSymbol(DataRefImpl &DRI) const;
  void getSection(DataRefImpl DRI, InMemoryStruct<macho::Section> &Res) const;
};

ObjectFile *ObjectFile::createMachOObjectFile(MemoryBuffer *Buffer) {
  std::string Err;
  MachOObject *MachOObj = MachOObject::LoadFromBuffer(Buffer, &Err);
  if (!MachOObj)
    return NULL;
  return new MachOObjectFile(Buffer, MachOObj);
}

/*===-- Symbols -----------------------------------------------------------===*/

void MachOObjectFile::moveToNextSymbol(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = MachOObj->getHeader().NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    if (LCI.Command.Type == macho::LCT_Symtab) {
      InMemoryStruct<macho::SymtabLoadCommand> SymtabLoadCmd;
      MachOObj->ReadSymtabLoadCommand(LCI, SymtabLoadCmd);
      if (DRI.d.b < SymtabLoadCmd->NumSymbolTableEntries)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

void MachOObjectFile::getSymbolTableEntry(DataRefImpl DRI,
    InMemoryStruct<macho::SymbolTableEntry> &Res) const {
  InMemoryStruct<macho::SymtabLoadCommand> SymtabLoadCmd;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSymtabLoadCommand(LCI, SymtabLoadCmd);

  if (RegisteredStringTable != DRI.d.a) {
    MachOObj->RegisterStringTable(*SymtabLoadCmd);
    RegisteredStringTable = DRI.d.a;
  }

  MachOObj->ReadSymbolTableEntry(SymtabLoadCmd->SymbolTableOffset, DRI.d.b,
                                 Res);
}


SymbolRef MachOObjectFile::getSymbolNext(DataRefImpl DRI) const {
  DRI.d.b++;
  moveToNextSymbol(DRI);
  return SymbolRef(DRI, this);
}

StringRef MachOObjectFile::getSymbolName(DataRefImpl DRI) const {
  InMemoryStruct<macho::SymbolTableEntry> Entry;
  getSymbolTableEntry(DRI, Entry);
  return MachOObj->getStringAtIndex(Entry->StringIndex);
}

uint64_t MachOObjectFile::getSymbolAddress(DataRefImpl DRI) const {
  InMemoryStruct<macho::SymbolTableEntry> Entry;
  getSymbolTableEntry(DRI, Entry);
  return Entry->Value;
}

uint64_t MachOObjectFile::getSymbolSize(DataRefImpl DRI) const {
  return UnknownAddressOrSize;
}

char MachOObjectFile::getSymbolNMTypeChar(DataRefImpl DRI) const {
  InMemoryStruct<macho::SymbolTableEntry> Entry;
  getSymbolTableEntry(DRI, Entry);

  char Char;
  switch (Entry->Type & macho::STF_TypeMask) {
    case macho::STT_Undefined:
      Char = 'u';
      break;
    case macho::STT_Absolute:
    case macho::STT_Section:
      Char = 's';
      break;
    default:
      Char = '?';
      break;
  }

  if (Entry->Flags & (macho::STF_External | macho::STF_PrivateExtern))
    Char = toupper(Char);
  return Char;
}

bool MachOObjectFile::isSymbolInternal(DataRefImpl DRI) const {
  InMemoryStruct<macho::SymbolTableEntry> Entry;
  getSymbolTableEntry(DRI, Entry);
  return Entry->Flags & macho::STF_StabsEntryMask;
}

ObjectFile::symbol_iterator MachOObjectFile::begin_symbols() const {
  // DRI.d.a = segment number; DRI.d.b = symbol index.
  DataRefImpl DRI;
  DRI.d.a = DRI.d.b = 0;
  moveToNextSymbol(DRI);
  return symbol_iterator(SymbolRef(DRI, this));
}

ObjectFile::symbol_iterator MachOObjectFile::end_symbols() const {
  DataRefImpl DRI;
  DRI.d.a = MachOObj->getHeader().NumLoadCommands;
  DRI.d.b = 0;
  return symbol_iterator(SymbolRef(DRI, this));
}


/*===-- Sections ----------------------------------------------------------===*/

void MachOObjectFile::moveToNextSection(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = MachOObj->getHeader().NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    if (LCI.Command.Type == macho::LCT_Segment) {
      InMemoryStruct<macho::SegmentLoadCommand> SegmentLoadCmd;
      MachOObj->ReadSegmentLoadCommand(LCI, SegmentLoadCmd);
      if (DRI.d.b < SegmentLoadCmd->NumSections)
        return;
    } else if (LCI.Command.Type == macho::LCT_Segment64) {
      InMemoryStruct<macho::Segment64LoadCommand> Segment64LoadCmd;
      MachOObj->ReadSegment64LoadCommand(LCI, Segment64LoadCmd);
      if (DRI.d.b < Segment64LoadCmd->NumSections)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

SectionRef MachOObjectFile::getSectionNext(DataRefImpl DRI) const {
  DRI.d.b++;
  moveToNextSection(DRI);
  return SectionRef(DRI, this);
}

void
MachOObjectFile::getSection(DataRefImpl DRI,
                            InMemoryStruct<macho::Section> &Res) const {
  InMemoryStruct<macho::SegmentLoadCommand> SLC;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSegmentLoadCommand(LCI, SLC);
  MachOObj->ReadSection(LCI, DRI.d.b, Res);
}

StringRef MachOObjectFile::getSectionName(DataRefImpl DRI) const {
  InMemoryStruct<macho::SegmentLoadCommand> SLC;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSegmentLoadCommand(LCI, SLC);
  InMemoryStruct<macho::Section> Sect;
  MachOObj->ReadSection(LCI, DRI.d.b, Sect);

  static char Result[34];
  strcpy(Result, SLC->Name);
  strcat(Result, ",");
  strcat(Result, Sect->Name);
  return StringRef(Result);
}

uint64_t MachOObjectFile::getSectionAddress(DataRefImpl DRI) const {
  InMemoryStruct<macho::Section> Sect;
  getSection(DRI, Sect);
  return Sect->Address;
}

uint64_t MachOObjectFile::getSectionSize(DataRefImpl DRI) const {
  InMemoryStruct<macho::Section> Sect;
  getSection(DRI, Sect);
  return Sect->Size;
}

StringRef MachOObjectFile::getSectionContents(DataRefImpl DRI) const {
  InMemoryStruct<macho::Section> Sect;
  getSection(DRI, Sect);
  return MachOObj->getData(Sect->Offset, Sect->Size);
}

bool MachOObjectFile::isSectionText(DataRefImpl DRI) const {
  InMemoryStruct<macho::SegmentLoadCommand> SLC;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSegmentLoadCommand(LCI, SLC);
  return !strcmp(SLC->Name, "__TEXT");
}

ObjectFile::section_iterator MachOObjectFile::begin_sections() const {
  DataRefImpl DRI;
  DRI.d.a = DRI.d.b = 0;
  moveToNextSection(DRI);
  return section_iterator(SectionRef(DRI, this));
}

ObjectFile::section_iterator MachOObjectFile::end_sections() const {
  DataRefImpl DRI;
  DRI.d.a = MachOObj->getHeader().NumLoadCommands;
  DRI.d.b = 0;
  return section_iterator(SectionRef(DRI, this));
}

/*===-- Miscellaneous -----------------------------------------------------===*/

uint8_t MachOObjectFile::getBytesInAddress() const {
  return MachOObj->is64Bit() ? 8 : 4;
}

StringRef MachOObjectFile::getFileFormatName() const {
  if (!MachOObj->is64Bit()) {
    switch (MachOObj->getHeader().CPUType) {
    case 0x00000007:
      return "MACHO32-i386";
    case 0x01000007:
      return "MACHO32-x86-64";
    case 0x0000000c:
      return "MACHO32-arm";
    case 0x00000012:
      return "MACHO32-ppc";
    case 0x01000012:
      return "MACHO32-ppc64";
    }
  }

  switch (MachOObj->getHeader().CPUType) {
  case 0x00000007:
    return "MACHO64-i386";
  case 0x01000007:
    return "MACHO64-x86-64";
  case 0x0000000c:
    return "MACHO64-arm";
  case 0x00000012:
    return "MACHO64-ppc";
  case 0x01000012:
    return "MACHO64-ppc64";
  default:
    return "MACHO64-unknown";
  }
}

unsigned MachOObjectFile::getArch() const {
  switch (MachOObj->getHeader().CPUType) {
  case 0x00000007:
    return Triple::x86;
  case 0x01000007:
    return Triple::x86_64;
  case 0x0000000c:
    return Triple::arm;
  case 0x00000012:
    return Triple::ppc;
  case 0x01000012:
    return Triple::ppc64;
  default:
    return Triple::UnknownArch;
  }
}

} // end namespace llvm

