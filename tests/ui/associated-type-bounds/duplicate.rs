#![feature(associated_type_bounds)]
#![feature(type_alias_impl_trait)]

use std::iter;
use std::mem::ManuallyDrop;

struct SI1<T: Iterator<Item: Copy, Item: Send>> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    f: T,
}
struct SI2<T: Iterator<Item: Copy, Item: Copy>> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    f: T,
}
struct SI3<T: Iterator<Item: 'static, Item: 'static>> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    f: T,
}
struct SW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
    f: T,
}
struct SW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
    f: T,
}
struct SW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
    f: T,
}

enum EI1<T: Iterator<Item: Copy, Item: Send>> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    V(T),
}
enum EI2<T: Iterator<Item: Copy, Item: Copy>> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    V(T),
}
enum EI3<T: Iterator<Item: 'static, Item: 'static>> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    V(T),
}
enum EW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
    V(T),
}
enum EW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
    V(T),
}
enum EW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
    V(T),
}

union UI1<T: Iterator<Item: Copy, Item: Send>> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    f: ManuallyDrop<T>,
}
union UI2<T: Iterator<Item: Copy, Item: Copy>> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    f: ManuallyDrop<T>,
}
union UI3<T: Iterator<Item: 'static, Item: 'static>> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    f: ManuallyDrop<T>,
}
union UW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
    f: ManuallyDrop<T>,
}
union UW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
    f: ManuallyDrop<T>,
}
union UW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
    f: ManuallyDrop<T>,
}

fn FI1<T: Iterator<Item: Copy, Item: Send>>() {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
fn FI2<T: Iterator<Item: Copy, Item: Copy>>() {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
fn FI3<T: Iterator<Item: 'static, Item: 'static>>() {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
fn FW1<T>()
where
    T: Iterator<Item: Copy, Item: Send>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
}
fn FW2<T>()
where
    T: Iterator<Item: Copy, Item: Copy>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
}
fn FW3<T>()
where
    T: Iterator<Item: 'static, Item: 'static>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
}

fn FRPIT1() -> impl Iterator<Item: Copy, Item: Send> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    iter::empty()
}
fn FRPIT2() -> impl Iterator<Item: Copy, Item: Copy> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    iter::empty()
}
fn FRPIT3() -> impl Iterator<Item: 'static, Item: 'static> {
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    iter::empty()
}
fn FAPIT1(_: impl Iterator<Item: Copy, Item: Send>) {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
fn FAPIT2(_: impl Iterator<Item: Copy, Item: Copy>) {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
fn FAPIT3(_: impl Iterator<Item: 'static, Item: 'static>) {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]

type TAI1<T: Iterator<Item: Copy, Item: Send>> = T;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type TAI2<T: Iterator<Item: Copy, Item: Copy>> = T;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type TAI3<T: Iterator<Item: 'static, Item: 'static>> = T;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type TAW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
= T;
type TAW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
= T;
type TAW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
= T;

type ETAI1<T: Iterator<Item: Copy, Item: Send>> = impl Copy;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type ETAI2<T: Iterator<Item: Copy, Item: Copy>> = impl Copy;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type ETAI3<T: Iterator<Item: 'static, Item: 'static>> = impl Copy;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type ETAI4 = impl Iterator<Item: Copy, Item: Send>;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type ETAI5 = impl Iterator<Item: Copy, Item: Copy>;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type ETAI6 = impl Iterator<Item: 'static, Item: 'static>;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]

trait TRI1<T: Iterator<Item: Copy, Item: Send>> {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
trait TRI2<T: Iterator<Item: Copy, Item: Copy>> {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
trait TRI3<T: Iterator<Item: 'static, Item: 'static>> {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
trait TRS1: Iterator<Item: Copy, Item: Send> {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
//~| ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
trait TRS2: Iterator<Item: Copy, Item: Copy> {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
//~| ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
trait TRS3: Iterator<Item: 'static, Item: 'static> {}
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
//~| ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
trait TRW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
}
trait TRW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
}
trait TRW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
}
trait TRSW1
where
    Self: Iterator<Item: Copy, Item: Send>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    //~| ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
}
trait TRSW2
where
    Self: Iterator<Item: Copy, Item: Copy>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    //~| ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
}
trait TRSW3
where
    Self: Iterator<Item: 'static, Item: 'static>,
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
    //~| ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
{
}
trait TRA1 {
    type A: Iterator<Item: Copy, Item: Send>;
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
}
trait TRA2 {
    type A: Iterator<Item: Copy, Item: Copy>;
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
}
trait TRA3 {
    type A: Iterator<Item: 'static, Item: 'static>;
    //~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
}

type TADyn1 = dyn Iterator<Item: Copy, Item: Send>;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type TADyn2 = Box<dyn Iterator<Item: Copy, Item: Copy>>;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]
type TADyn3 = dyn Iterator<Item: 'static, Item: 'static>;
//~^ ERROR the value of the associated type `Item` in trait `Iterator` is already specified [E0719]

fn main() {}
