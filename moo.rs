use std::fmt::Debug;
use std::marker::PhantomData;
use std::slice;
use std::fmt;

pub trait FileHeader: Debug + Pod {
    // Ideally this would be a `u64: From<Word>`, but can't express that.
    type Word: Into<u64> + Default + Copy;
    type Sword: Into<i64>;
    type Endian: Endian;
    type ProgramHeader: ProgramHeader<Elf = Self, Endian = Self::Endian, Word = Self::Word>;
    type SectionHeader: SectionHeader<Elf = Self, Endian = Self::Endian, Word = Self::Word>;
    type CompressionHeader: CompressionHeader<Endian = Self::Endian, Word = Self::Word>;
    type NoteHeader: NoteHeader<Endian = Self::Endian>;
    type Dyn: Dyn<Endian = Self::Endian, Word = Self::Word>;
    type Sym: Sym<Endian = Self::Endian, Word = Self::Word>;
    type Rel: Rel<Endian = Self::Endian, Word = Self::Word>;
    type Rela: Rela<Endian = Self::Endian, Word = Self::Word> + From<Self::Rel>;
    type Relr: Relr<Endian = Self::Endian, Word = Self::Word>;
}

pub unsafe trait Pod: Copy + 'static {}

pub trait ProgramHeader: Debug + Pod {
    type Elf: FileHeader<ProgramHeader = Self, Endian = Self::Endian, Word = Self::Word>;
    type Word: Into<u64>;
    type Endian: Endian;
}

pub trait SectionHeader: Debug + Pod {
    type Elf: FileHeader<SectionHeader = Self, Endian = Self::Endian, Word = Self::Word>;
    type Word: Into<u64>;
    type Endian: Endian;
}

pub trait CompressionHeader: Debug + Pod {
    type Word: Into<u64>;
    type Endian: Endian;
}

pub trait NoteHeader: Debug + Pod {
    type Endian: Endian;
}

pub trait Dyn: Debug + Pod {
    type Word: Into<u64>;
    type Endian: Endian;
}

pub trait Sym: Debug + Pod {
    type Word: Into<u64>;
    type Endian: Endian;
}

pub trait Rel: Debug + Pod + Clone {
    type Word: Into<u64>;
    type Sword: Into<i64>;
    type Endian: Endian;
}

pub trait Rela: Debug + Pod + Clone {
    type Word: Into<u64>;
    type Sword: Into<i64>;
    type Endian: Endian;
}

pub trait Relr: Debug + Pod + Clone {
    type Word: Into<u64>;
    type Endian: Endian;
}

pub trait ReadRef<'a>: Clone + Copy {}

pub struct SectionIndex(pub u32);

pub struct ElfFile<'data, Elf, R = &'data [u8]>
where
    Elf: FileHeader,
    R: ReadRef<'data>,
{
    pub endian: Elf::Endian,
    pub data: R,
    pub header: &'data Elf,
    pub segments: &'data [Elf::ProgramHeader],
    pub sections: SectionTable<'data, Elf, R>,
    pub relocations: RelocationSections,
    pub symbols: SymbolTable<'data, Elf, R>,
    pub dynamic_symbols: SymbolTable<'data, Elf, R>,
}

pub struct SymbolIndex(pub usize);

pub struct SymbolTable<'data, Elf: FileHeader, R = &'data [u8]>
where
    R: ReadRef<'data>,
{
    section: SectionIndex,
    string_section: SectionIndex,
    shndx_section: SectionIndex,
    symbols: &'data [Elf::Sym],
    strings: StringTable<'data, R>,
    shndx: &'data [U32<Elf::Endian>],
}

pub trait Endian: Debug + Default + Clone + Copy + PartialEq + Eq + 'static {}

pub type U32<E> = U32Bytes<E>;

#[repr(transparent)]
pub struct U32Bytes<E: Endian>([u8; 4], PhantomData<E>);

impl<E: Endian> fmt::Debug for U32Bytes<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

pub struct RelocationSections {
    relocations: Vec<usize>,
}

pub struct SectionTable<'data, Elf: FileHeader, R = &'data [u8]>
where
    R: ReadRef<'data>,
{
    sections: &'data [Elf::SectionHeader],
    strings: StringTable<'data, R>,
}

pub struct StringTable<'data, R = &'data [u8]>
where
    R: ReadRef<'data>,
{
    data: Option<R>,
    start: u64,
    end: u64,
    marker: PhantomData<&'data ()>,
}

pub enum ElfRelocationIterator<'data, Elf: FileHeader> {
    Rel(slice::Iter<'data, Elf::Rel>, Elf::Endian),
    Rela(slice::Iter<'data, Elf::Rela>, Elf::Endian, bool),
    Crel(CrelIterator<'data>),
}

struct CrelIteratorHeader {
    /// The number of encoded relocations.
    count: usize,
    /// The number of flag bits each relocation uses.
    flag_bits: u64,
    /// Shift of the relocation value.
    shift: u64,
    /// True if the relocation format encodes addend.
    is_rela: bool,
}

struct CrelIteratorState {
    /// Index of the current relocation.
    index: usize,
    /// Offset of the latest relocation.
    offset: u64,
    /// Addend of the latest relocation.
    addend: i64,
    /// Symbol index of the latest relocation.
    symidx: u32,
    /// Type of the latest relocation.
    typ: u32,
}

pub struct Bytes<'data>(pub &'data [u8]);

/// Compact relocation iterator.
pub struct CrelIterator<'data> {
    /// Input stream reader.
    data: Bytes<'data>,
    /// Parsed header information.
    header: CrelIteratorHeader,
    /// State of the iterator.
    state: CrelIteratorState,
}

pub struct ElfDynamicRelocationIterator<'data, 'file, Elf, R = &'data [u8]>
where
    Elf: FileHeader,
    R: ReadRef<'data>,
{
    /// The current relocation section index.
    pub section_index: SectionIndex,
    pub file: &'file ElfFile<'data, Elf, R>,
    pub relocations: Option<ElfRelocationIterator<'data, Elf>>,
}

#[non_exhaustive]
pub enum RelocationTarget {
    /// The target is a symbol.
    Symbol(SymbolIndex),
    /// The target is a section.
    Section(SectionIndex),
    /// The offset is an absolute address.
    Absolute,
}

#[non_exhaustive]
pub enum RelocationFlags {
    /// Format independent representation.
    Generic {
        /// The operation used to calculate the result of the relocation.
        kind: RelocationKind,
        /// Information about how the result of the relocation operation is encoded in the place.
        encoding: RelocationEncoding,
        /// The size in bits of the place of relocation.
        size: u8,
    },
    /// ELF relocation fields.
    Elf {
        /// `r_type` field in the ELF relocation.
        r_type: u32,
    },
    /// Mach-O relocation fields.
    MachO {
        /// `r_type` field in the Mach-O relocation.
        r_type: u8,
        /// `r_pcrel` field in the Mach-O relocation.
        r_pcrel: bool,
        /// `r_length` field in the Mach-O relocation.
        r_length: u8,
    },
    /// COFF relocation fields.
    Coff {
        /// `typ` field in the COFF relocation.
        typ: u16,
    },
    /// XCOFF relocation fields.
    Xcoff {
        /// `r_rtype` field in the XCOFF relocation.
        r_rtype: u8,
        /// `r_rsize` field in the XCOFF relocation.
        r_rsize: u8,
    },
}

#[non_exhaustive]
pub enum RelocationEncoding {
    /// The relocation encoding is unknown.
    Unknown,
    /// Generic encoding.
    Generic,

    /// x86 sign extension at runtime.
    ///
    /// Used with `RelocationKind::Absolute`.
    X86Signed,
    /// x86 rip-relative addressing.
    ///
    /// The `RelocationKind` must be PC relative.
    X86RipRelative,
    /// x86 rip-relative addressing in movq instruction.
    ///
    /// The `RelocationKind` must be PC relative.
    X86RipRelativeMovq,
    /// x86 branch instruction.
    ///
    /// The `RelocationKind` must be PC relative.
    X86Branch,

    /// s390x PC-relative offset shifted right by one bit.
    ///
    /// The `RelocationKind` must be PC relative.
    S390xDbl,

    /// AArch64 call target.
    ///
    /// The `RelocationKind` must be PC relative.
    AArch64Call,

    /// LoongArch branch offset with two trailing zeros.
    ///
    /// The `RelocationKind` must be PC relative.
    LoongArchBranch,

    /// SHARC+ 48-bit Type A instruction
    ///
    /// Represents these possible variants, each with a corresponding
    /// `R_SHARC_*` constant:
    ///
    /// * 24-bit absolute address
    /// * 32-bit absolute address
    /// * 6-bit relative address
    /// * 24-bit relative address
    /// * 6-bit absolute address in the immediate value field
    /// * 16-bit absolute address in the immediate value field
    SharcTypeA,

    /// SHARC+ 32-bit Type B instruction
    ///
    /// Represents these possible variants, each with a corresponding
    /// `R_SHARC_*` constant:
    ///
    /// * 6-bit absolute address in the immediate value field
    /// * 7-bit absolute address in the immediate value field
    /// * 16-bit absolute address
    /// * 6-bit relative address
    SharcTypeB,

    /// E2K 64-bit value stored in two LTS
    ///
    /// Memory representation:
    /// ```text
    /// 0: LTS1 = value[63:32]
    /// 4: LTS0 = value[31:0]
    /// ```
    E2KLit,

    /// E2K 28-bit value stored in CS0
    E2KDisp,
}

#[non_exhaustive]
pub enum RelocationKind {
    /// The operation is unknown.
    Unknown,
    /// S + A
    Absolute,
    /// S + A - P
    Relative,
    /// G + A - GotBase
    Got,
    /// G + A - P
    GotRelative,
    /// GotBase + A - P
    GotBaseRelative,
    /// S + A - GotBase
    GotBaseOffset,
    /// L + A - P
    PltRelative,
    /// S + A - Image
    ImageOffset,
    /// S + A - Section
    SectionOffset,
    /// The index of the section containing the symbol.
    SectionIndex,
}

pub struct Relocation {
    kind: RelocationKind,
    encoding: RelocationEncoding,
    size: u8,
    target: RelocationTarget,
    addend: i64,
    implicit_addend: bool,
    flags: RelocationFlags,
}


impl<'data, 'file, Elf, R> Iterator for ElfDynamicRelocationIterator<'data, 'file, Elf, R>
where
    Elf: FileHeader,
    R: ReadRef<'data>,
{
    type Item = (u64, Relocation);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

fn main() {}