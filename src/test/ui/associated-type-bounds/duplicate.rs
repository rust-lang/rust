// ignore-tidy-linelength

#![feature(associated_type_bounds)]
#![feature(type_alias_impl_trait)]
#![feature(impl_trait_in_bindings)] //~ WARN the feature `impl_trait_in_bindings` is incomplete
#![feature(untagged_unions)]

use std::iter;

struct SI1<T: Iterator<Item: Copy, Item: Send>> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
struct SI2<T: Iterator<Item: Copy, Item: Copy>> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
struct SI3<T: Iterator<Item: 'static, Item: 'static>> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
struct SW1<T> where T: Iterator<Item: Copy, Item: Send> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
struct SW2<T> where T: Iterator<Item: Copy, Item: Copy> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
struct SW3<T> where T: Iterator<Item: 'static, Item: 'static> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]

enum EI1<T: Iterator<Item: Copy, Item: Send>> { V(T) }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
enum EI2<T: Iterator<Item: Copy, Item: Copy>> { V(T) }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
enum EI3<T: Iterator<Item: 'static, Item: 'static>> { V(T) }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
enum EW1<T> where T: Iterator<Item: Copy, Item: Send> { V(T) }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
enum EW2<T> where T: Iterator<Item: Copy, Item: Copy> { V(T) }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
enum EW3<T> where T: Iterator<Item: 'static, Item: 'static> { V(T) }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]

union UI1<T: Iterator<Item: Copy, Item: Send>> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
union UI2<T: Iterator<Item: Copy, Item: Copy>> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
union UI3<T: Iterator<Item: 'static, Item: 'static>> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
union UW1<T> where T: Iterator<Item: Copy, Item: Send> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
union UW2<T> where T: Iterator<Item: Copy, Item: Copy> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
union UW3<T> where T: Iterator<Item: 'static, Item: 'static> { f: T }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]

fn FI1<T: Iterator<Item: Copy, Item: Send>>() {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FI2<T: Iterator<Item: Copy, Item: Copy>>() {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FI3<T: Iterator<Item: 'static, Item: 'static>>() {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FW1<T>() where T: Iterator<Item: Copy, Item: Send> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FW2<T>() where T: Iterator<Item: Copy, Item: Copy> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FW3<T>() where T: Iterator<Item: 'static, Item: 'static> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]

fn FRPIT1() -> impl Iterator<Item: Copy, Item: Send> { iter::empty() }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FRPIT2() -> impl Iterator<Item: Copy, Item: Copy> { iter::empty() }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FRPIT3() -> impl Iterator<Item: 'static, Item: 'static> { iter::empty() }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FAPIT1(_: impl Iterator<Item: Copy, Item: Send>) {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FAPIT2(_: impl Iterator<Item: Copy, Item: Copy>) {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn FAPIT3(_: impl Iterator<Item: 'static, Item: 'static>) {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]

const CIT1: impl Iterator<Item: Copy, Item: Send> = iter::empty();
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
const CIT2: impl Iterator<Item: Copy, Item: Copy> = iter::empty();
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
const CIT3: impl Iterator<Item: 'static, Item: 'static> = iter::empty();
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
static SIT1: impl Iterator<Item: Copy, Item: Send> = iter::empty();
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
static SIT2: impl Iterator<Item: Copy, Item: Copy> = iter::empty();
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
static SIT3: impl Iterator<Item: 'static, Item: 'static> = iter::empty();
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]

fn lit1() { let _: impl Iterator<Item: Copy, Item: Send> = iter::empty(); }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn lit2() { let _: impl Iterator<Item: Copy, Item: Copy> = iter::empty(); }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
fn lit3() { let _: impl Iterator<Item: 'static, Item: 'static> = iter::empty(); }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]

type TAI1<T: Iterator<Item: Copy, Item: Send>> = T;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
type TAI2<T: Iterator<Item: Copy, Item: Copy>> = T;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
type TAI3<T: Iterator<Item: 'static, Item: 'static>> = T;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
type TAW1<T> where T: Iterator<Item: Copy, Item: Send> = T;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
type TAW2<T> where T: Iterator<Item: Copy, Item: Copy> = T;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
type TAW3<T> where T: Iterator<Item: 'static, Item: 'static> = T;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]

type ETAI1<T: Iterator<Item: Copy, Item: Send>> = impl Copy;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR could not find defining uses
type ETAI2<T: Iterator<Item: Copy, Item: Copy>> = impl Copy;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR could not find defining uses
type ETAI3<T: Iterator<Item: 'static, Item: 'static>> = impl Copy;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR could not find defining uses
type ETAI4 = impl Iterator<Item: Copy, Item: Send>;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR could not find defining uses
//~| ERROR could not find defining uses
//~| ERROR could not find defining uses
type ETAI5 = impl Iterator<Item: Copy, Item: Copy>;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR could not find defining uses
//~| ERROR could not find defining uses
//~| ERROR could not find defining uses
type ETAI6 = impl Iterator<Item: 'static, Item: 'static>;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR could not find defining uses
//~| ERROR could not find defining uses
//~| ERROR could not find defining uses

trait TRI1<T: Iterator<Item: Copy, Item: Send>> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRI2<T: Iterator<Item: Copy, Item: Copy>> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRI3<T: Iterator<Item: 'static, Item: 'static>> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRS1: Iterator<Item: Copy, Item: Send> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRS2: Iterator<Item: Copy, Item: Copy> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRS3: Iterator<Item: 'static, Item: 'static> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRW1<T> where T: Iterator<Item: Copy, Item: Send> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRW2<T> where T: Iterator<Item: Copy, Item: Copy> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRW3<T> where T: Iterator<Item: 'static, Item: 'static> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRSW1 where Self: Iterator<Item: Copy, Item: Send> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRSW2 where Self: Iterator<Item: Copy, Item: Copy> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRSW3 where Self: Iterator<Item: 'static, Item: 'static> {}
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRA1 { type A: Iterator<Item: Copy, Item: Send>; }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRA2 { type A: Iterator<Item: Copy, Item: Copy>; }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
trait TRA3 { type A: Iterator<Item: 'static, Item: 'static>; }
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]

type TADyn1 = dyn Iterator<Item: Copy, Item: Send>;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR could not find defining uses
//~| ERROR could not find defining uses
type TADyn2 = Box<dyn Iterator<Item: Copy, Item: Copy>>;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR could not find defining uses
//~| ERROR could not find defining uses
type TADyn3 = dyn Iterator<Item: 'static, Item: 'static>;
//~^ ERROR the value of the associated type `Item` (from trait `std::iter::Iterator`) is already specified [E0719]
//~| ERROR could not find defining uses
//~| ERROR could not find defining uses

fn main() {}
