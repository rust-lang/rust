// only-cdb
// ignore-tidy-linelength
// compile-flags:-g

// This started failing recently. See https://github.com/rust-lang/rust/issues/88796
// FIXME: fix and unignore this
// ignore-windows

// cdb-command: g

// Note: The natvis used to visualize niche-layout enums don't work correctly in cdb
//       so the best we can do is to make sure we are generating the right debuginfo.
//       Therefore, we use the `!` [format specifier](https://docs.microsoft.com/en-us/visualstudio/debugger/format-specifiers-in-cpp?view=vs-2019#BKMK_Visual_Studio_2012_format_specifiers)
//       to disable the natvis for a given expression. We also provide the `-r2` flag
//       to expand the expression 2 levels.

// cdb-command: dx -r2 a,!
// cdb-check:a,!              [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>]
// cdb-check:    [+0x000] dataful_variant  [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>::Some]
// cdb-check:        [+0x000] __0              : Low (0x2) [Type: msvc_pretty_enums::CStyleEnum]
// cdb-check:    [+0x000] discriminant     : 0x2 [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>::Discriminant$]

// cdb-command: dx -r2 b,!
// cdb-check:b,!              [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>]
// cdb-check:    [+0x000] dataful_variant  [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>::Some]
// cdb-check:        [+0x000] __0              : 0x11 [Type: msvc_pretty_enums::CStyleEnum]
// cdb-check:    [+0x000] discriminant     : None (0x11) [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>::Discriminant$]

// cdb-command: dx -r2 c,!
// cdb-check:c,!              [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>]
// cdb-check:    [+0x000] dataful_variant  [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>::Data]
// cdb-check:        [+0x000] my_data          : 0x11 [Type: msvc_pretty_enums::CStyleEnum]
// cdb-check:    [+0x000] discriminant     : Tag1 (0x11) [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>::Discriminant$]

// cdb-command: dx -r2 d,!
// cdb-check:d,!              [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>]
// cdb-check:    [+0x000] dataful_variant  [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>::Data]
// cdb-check:        [+0x000] my_data          : High (0x10) [Type: msvc_pretty_enums::CStyleEnum]
// cdb-check:    [+0x000] discriminant     : 0x10 [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>::Discriminant$]

// cdb-command: dx -r2 e,!
// cdb-check:e,!              [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>]
// cdb-check:    [+0x000] dataful_variant  [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>::Data]
// cdb-check:        [+0x000] my_data          : 0x13 [Type: msvc_pretty_enums::CStyleEnum]
// cdb-check:    [+0x000] discriminant     : Tag2 (0x13) [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>::Discriminant$]

// cdb-command: dx -r2 f,!
// cdb-check:f,!              [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>]
// cdb-check:    [+0x000] dataful_variant  [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>::Some]
// cdb-check:        [+0x000] __0              : 0x[...] : 0x1 [Type: unsigned int *]
// cdb-check:    [+0x000] discriminant     : 0x[...] [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>::Discriminant$]

// cdb-command: dx -r2 g,!
// cdb-check:g,!              [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>]
// cdb-check:    [+0x000] dataful_variant  [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>::Some]
// cdb-check:        [+0x000] __0              : 0x0 [Type: unsigned int *]
// cdb-check:    [+0x000] discriminant     : None (0x0) [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>::Discriminant$]

// cdb-command: dx -r2 h,!
// cdb-check:h,!              : Some [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [+0x000] variant0         [Type: enum$<core::option::Option<u32> >::None]
// cdb-check:    [+0x000] variant1         [Type: enum$<core::option::Option<u32> >::Some]
// cdb-check:        [+0x004] __0              : 0xc [Type: unsigned int]
// cdb-check:    [+0x000] discriminant     : Some (0x1) [Type: core::option::Option]

// cdb-command: dx h
// cdb-check:h                : Some [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [<Raw View>]     [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [variant]        : Some
// cdb-check:    [+0x004] __0              : 0xc [Type: unsigned int]

// cdb-command: dx -r2 i,!
// cdb-check:i,!              : None [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [+0x000] variant0         [Type: enum$<core::option::Option<u32> >::None]
// cdb-check:    [+0x000] variant1         [Type: enum$<core::option::Option<u32> >::Some]
// cdb-check:        [+0x004] __0              : 0x[...] [Type: unsigned int]
// cdb-check:    [+0x000] discriminant     : None (0x0) [Type: core::option::Option]

// cdb-command: dx i
// cdb-check:i                : None [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [<Raw View>]     [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [variant]        : None

// cdb-command: dx j
// cdb-check:j                : High (0x10) [Type: msvc_pretty_enums::CStyleEnum]

// cdb-command: dx -r2 k,!
// cdb-check:k,!              [Type: enum$<core::option::Option<alloc::string::String>, 1, [...], Some>]
// cdb-check:    [+0x000] dataful_variant  [Type: enum$<core::option::Option<alloc::string::String>, 1, [...], Some>::Some]
// cdb-check:        [+0x000] __0              [Type: alloc::string::String]
// cdb-check:    [+0x000] discriminant     : 0x[...] [Type: enum$<core::option::Option<alloc::string::String>, 1, [...], Some>::Discriminant$]

// cdb-command: dx -r2 l,!
// cdb-check:l,!              : $T2 [Type: enum$<core::result::Result<u32,enum$<msvc_pretty_enums::Empty> >, Ok>]
// cdb-check:    [+0x000] Ok               [Type: enum$<core::result::Result<u32,enum$<msvc_pretty_enums::Empty> >, Ok>::Ok]
// cdb-check:        [+0x000] __0              : 0x2a [Type: unsigned int]

pub enum CStyleEnum {
    Low = 2,
    High = 16,
}

pub enum NicheLayoutEnum {
    Tag1,
    Data { my_data: CStyleEnum },
    Tag2,
}

pub enum Empty { }

fn main() {
    let a = Some(CStyleEnum::Low);
    let b = Option::<CStyleEnum>::None;
    let c = NicheLayoutEnum::Tag1;
    let d = NicheLayoutEnum::Data { my_data: CStyleEnum::High };
    let e = NicheLayoutEnum::Tag2;
    let f = Some(&1u32);
    let g = Option::<&'static u32>::None;
    let h = Some(12u32);
    let i = Option::<u32>::None;
    let j = CStyleEnum::High;
    let k = Some("IAMA optional string!".to_string());
    let l = Result::<u32, Empty>::Ok(42);

    zzz(); // #break
}

fn zzz() { () }
