// compile-fail
// ignore-tidy-linelength
// error-pattern:could not find defining uses

#![feature(associated_type_bounds)]
#![feature(existential_type)]
#![feature(impl_trait_in_bindings)]
#![feature(untagged_unions)]

use std::iter;

struct SI1<T: Iterator<Item: Copy, Item: Send>> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
struct SI2<T: Iterator<Item: Copy, Item: Copy>> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
struct SI3<T: Iterator<Item: 'static, Item: 'static>> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
struct SW1<T> where T: Iterator<Item: Copy, Item: Send> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
struct SW2<T> where T: Iterator<Item: Copy, Item: Copy> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
struct SW3<T> where T: Iterator<Item: 'static, Item: 'static> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

enum EI1<T: Iterator<Item: Copy, Item: Send>> { V(T) }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
enum EI2<T: Iterator<Item: Copy, Item: Copy>> { V(T) }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
enum EI3<T: Iterator<Item: 'static, Item: 'static>> { V(T) }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
enum EW1<T> where T: Iterator<Item: Copy, Item: Send> { V(T) }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
enum EW2<T> where T: Iterator<Item: Copy, Item: Copy> { V(T) }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
enum EW3<T> where T: Iterator<Item: 'static, Item: 'static> { V(T) }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

union UI1<T: Iterator<Item: Copy, Item: Send>> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
union UI2<T: Iterator<Item: Copy, Item: Copy>> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
union UI3<T: Iterator<Item: 'static, Item: 'static>> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
union UW1<T> where T: Iterator<Item: Copy, Item: Send> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
union UW2<T> where T: Iterator<Item: Copy, Item: Copy> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
union UW3<T> where T: Iterator<Item: 'static, Item: 'static> { f: T }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

fn FI1<T: Iterator<Item: Copy, Item: Send>>() {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FI2<T: Iterator<Item: Copy, Item: Copy>>() {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FI3<T: Iterator<Item: 'static, Item: 'static>>() {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FW1<T>() where T: Iterator<Item: Copy, Item: Send> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FW2<T>() where T: Iterator<Item: Copy, Item: Copy> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FW3<T>() where T: Iterator<Item: 'static, Item: 'static> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

fn FRPIT1() -> impl Iterator<Item: Copy, Item: Send> { iter::empty() }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FRPIT2() -> impl Iterator<Item: Copy, Item: Copy> { iter::empty() }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FRPIT3() -> impl Iterator<Item: 'static, Item: 'static> { iter::empty() }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FAPIT1(_: impl Iterator<Item: Copy, Item: Send>) {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FAPIT2(_: impl Iterator<Item: Copy, Item: Copy>) {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn FAPIT3(_: impl Iterator<Item: 'static, Item: 'static>) {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

const CIT1: impl Iterator<Item: Copy, Item: Send> = iter::empty();
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
const CIT2: impl Iterator<Item: Copy, Item: Copy> = iter::empty();
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
const CIT3: impl Iterator<Item: 'static, Item: 'static> = iter::empty();
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
static SIT1: impl Iterator<Item: Copy, Item: Send> = iter::empty();
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
static SIT2: impl Iterator<Item: Copy, Item: Copy> = iter::empty();
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
static SIT3: impl Iterator<Item: 'static, Item: 'static> = iter::empty();
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

fn lit1() { let _: impl Iterator<Item: Copy, Item: Send> = iter::empty(); }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn lit2() { let _: impl Iterator<Item: Copy, Item: Copy> = iter::empty(); }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
fn lit3() { let _: impl Iterator<Item: 'static, Item: 'static> = iter::empty(); }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

type TAI1<T: Iterator<Item: Copy, Item: Send>> = T;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
type TAI2<T: Iterator<Item: Copy, Item: Copy>> = T;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
type TAI3<T: Iterator<Item: 'static, Item: 'static>> = T;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
type TAW1<T> where T: Iterator<Item: Copy, Item: Send> = T;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
type TAW2<T> where T: Iterator<Item: Copy, Item: Copy> = T;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
type TAW3<T> where T: Iterator<Item: 'static, Item: 'static> = T;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

existential type ETAI1<T: Iterator<Item: Copy, Item: Send>>: Copy;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
existential type ETAI2<T: Iterator<Item: Copy, Item: Copy>>: Copy;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
existential type ETAI3<T: Iterator<Item: 'static, Item: 'static>>: Copy;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
existential type ETAI4: Iterator<Item: Copy, Item: Send>;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
existential type ETAI5: Iterator<Item: Copy, Item: Copy>;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
existential type ETAI6: Iterator<Item: 'static, Item: 'static>;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

trait TRI1<T: Iterator<Item: Copy, Item: Send>> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRI2<T: Iterator<Item: Copy, Item: Copy>> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRI3<T: Iterator<Item: 'static, Item: 'static>> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRS1: Iterator<Item: Copy, Item: Send> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRS2: Iterator<Item: Copy, Item: Copy> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRS3: Iterator<Item: 'static, Item: 'static> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRW1<T> where T: Iterator<Item: Copy, Item: Send> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRW2<T> where T: Iterator<Item: Copy, Item: Copy> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRW3<T> where T: Iterator<Item: 'static, Item: 'static> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRSW1 where Self: Iterator<Item: Copy, Item: Send> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRSW2 where Self: Iterator<Item: Copy, Item: Copy> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRSW3 where Self: Iterator<Item: 'static, Item: 'static> {}
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRA1 { type A: Iterator<Item: Copy, Item: Send>; }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRA2 { type A: Iterator<Item: Copy, Item: Copy>; }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
trait TRA3 { type A: Iterator<Item: 'static, Item: 'static>; }
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

type TADyn1 = dyn Iterator<Item: Copy, Item: Send>;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
type TADyn2 = Box<dyn Iterator<Item: Copy, Item: Copy>>;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]
type TADyn3 = dyn Iterator<Item: 'static, Item: 'static>;
//~^ the value of the associated type `Item` (from the trait `std::iter::Iterator`) is already specified [E0719]

fn main() {}
