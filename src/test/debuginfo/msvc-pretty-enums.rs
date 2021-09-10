// only-cdb
// compile-flags:-g

// cdb-command: g

// cdb-command: dx a
// cdb-check:a                :  Some({...}) [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>]
// cdb-check:    [<Raw View>]     [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>]
// cdb-check:    [variant]        :  Some
// cdb-check:    [+0x000] __0              : Low (0x2) [Type: msvc_pretty_enums::CStyleEnum]

// cdb-command: dx b
// cdb-check:b                : None [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>]
// cdb-check:    [<Raw View>]     [Type: enum$<core::option::Option<enum$<msvc_pretty_enums::CStyleEnum> >, 2, 16, Some>]
// cdb-check:    [variant]        : None

// cdb-command: dx c
// cdb-check:c                : Tag1 [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>]
// cdb-check:    [<Raw View>]     [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>]
// cdb-check:    [variant]        : Tag1

// cdb-command: dx d
// cdb-check:d                :  Data({...}) [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>]
// cdb-check:    [<Raw View>]     [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>]
// cdb-check:    [variant]        :  Data
// cdb-check:    [+0x000] my_data          : High (0x10) [Type: msvc_pretty_enums::CStyleEnum]

// cdb-command: dx e
// cdb-check:e                : Tag2 [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>]
// cdb-check:    [<Raw View>]     [Type: enum$<msvc_pretty_enums::NicheLayoutEnum, 2, 16, Data>]
// cdb-check:    [variant]        : Tag2

// cdb-command: dx f
// cdb-check:f                :  Some({...}) [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>]
// cdb-check:    [<Raw View>]     [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>]
// cdb-check:    [variant]        :  Some
// cdb-check:    [+0x000] __0              : 0x[...] : 0x1 [Type: unsigned int *]

// cdb-command: dx g
// cdb-check:g                : None [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>]
// cdb-check:    [<Raw View>]     [Type: enum$<core::option::Option<ref$<u32> >, 1, [...], Some>]
// cdb-check:    [variant]        : None

// cdb-command: dx h
// cdb-check:h                : Some [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [<Raw View>]     [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [variant]        : Some
// cdb-check:    [+0x004] __0              : 0xc [Type: unsigned int]

// cdb-command: dx i
// cdb-check:i                : None [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [<Raw View>]     [Type: enum$<core::option::Option<u32> >]
// cdb-check:    [variant]        : None

// cdb-command: dx j
// cdb-check:j                : High (0x10) [Type: msvc_pretty_enums::CStyleEnum]

// cdb-command: dx k
// cdb-check:k                :  Some({...}) [Type: enum$<core::option::Option<alloc::string::String>, 1, [...], Some>]
// cdb-check:    [<Raw View>]     [Type: enum$<core::option::Option<alloc::string::String>, 1, [...], Some>]
// cdb-check:    [variant]        :  Some
// cdb-check:    [+0x000] __0              : "IAMA optional string!" [Type: alloc::string::String]

// cdb-command: dx l
// cdb-check:l                :  Ok [Type: enum$<core::result::Result<u32,enum$<msvc_pretty_enums::Empty> >, Ok>]
// cdb-check:    [<Raw View>]     [Type: enum$<core::result::Result<u32,enum$<msvc_pretty_enums::Empty> >, Ok>]
// cdb-check:    [variant]        :  Ok
// cdb-check:    [+0x000] __0              : 0x2a [Type: unsigned int]

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
