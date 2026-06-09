//@ revisions: edition2015 edition2021
//@ [edition2015] edition: 2015
//@ [edition2021] edition: 2021

#![feature(extern_types)]
#![feature(cfg_accessible)]

// Struct::unresolved - error

struct Struct {
    existing: u8,
}

#[cfg_accessible(Struct::existing)] //~ ERROR not sure
const A: bool = true;
#[cfg_accessible(Struct::unresolved)] //~ ERROR not sure
const B: bool = true;

// Union::unresolved - error

struct Union {
    existing: u8,
}

#[cfg_accessible(Union::existing)] //~ ERROR not sure
const A: bool = true;
#[cfg_accessible(Union::unresolved)] //~ ERROR not sure
const B: bool = true;

// Enum::unresolved - error

enum Enum {
    Existing { existing: u8 },
}

#[cfg_accessible(Enum::Existing::existing)] //~ ERROR not sure
const A: bool = true;
#[cfg_accessible(Enum::Existing::unresolved)] //~ ERROR not sure
const B: bool = true;
#[cfg_accessible(Enum::unresolved)] //~ ERROR not sure
const C: bool = true;

// Trait::unresolved - false or error, depending on edition (error if you can write Trait::foo
// instead of <dyn Trait>::foo for methods like impl dyn Trait { fn foo() {} })

trait Trait {}
impl dyn Trait { fn existing() {} }

// FIXME: Should be an error for edition > 2015
#[cfg_accessible(Trait::existing)] //~ ERROR not sure
const A: bool = true;
#[cfg_accessible(Trait::unresolved)] //~ ERROR not sure
const B: bool = true;

// TypeAlias::unresolved - error

type TypeAlias = Struct;

#[cfg_accessible(TypeAlias::existing)] //~ ERROR not sure
const A: bool = true;
#[cfg_accessible(TypeAlias::unresolved)] //~ ERROR not sure
const B: bool = true;

// ForeignType::unresolved - error

extern "C" {
    type ForeignType;
}

#[cfg_accessible(ForeignType::unresolved)] //~ ERROR not sure
const A: bool = true;

// AssocType::unresolved - error

trait AssocType {
    type AssocType;
}

#[cfg_accessible(AssocType::AssocType::unresolved)] //~ ERROR not sure
const A: bool = true;

// PrimitiveType::unresolved - error

#[cfg_accessible(u8::unresolved)] //~ ERROR not sure
const A: bool = true;
#[cfg_accessible(u8::is_ascii)] //~ ERROR not sure
const B: bool = true;

fn main() {}
