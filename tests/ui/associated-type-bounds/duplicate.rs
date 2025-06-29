//@ run-pass

#![feature(associated_const_equality, return_type_notation)]
#![allow(dead_code, refining_impl_trait_internal, type_alias_bounds)]

use std::iter;
use std::mem::ManuallyDrop;

struct SI1<T: Iterator<Item: Copy, Item: Send>> {
    f: T,
}
struct SI2<T: Iterator<Item: Copy, Item: Copy>> {
    f: T,
}
struct SI3<T: Iterator<Item: 'static, Item: 'static>> {
    f: T,
}
struct SW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
{
    f: T,
}
struct SW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
{
    f: T,
}
struct SW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
{
    f: T,
}

enum EI1<T: Iterator<Item: Copy, Item: Send>> {
    V(T),
}
enum EI2<T: Iterator<Item: Copy, Item: Copy>> {
    V(T),
}
enum EI3<T: Iterator<Item: 'static, Item: 'static>> {
    V(T),
}
enum EW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
{
    V(T),
}
enum EW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
{
    V(T),
}
enum EW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
{
    V(T),
}

union UI1<T: Iterator<Item: Copy, Item: Send>> {
    f: ManuallyDrop<T>,
}
union UI2<T: Iterator<Item: Copy, Item: Copy>> {
    f: ManuallyDrop<T>,
}
union UI3<T: Iterator<Item: 'static, Item: 'static>> {
    f: ManuallyDrop<T>,
}
union UW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
{
    f: ManuallyDrop<T>,
}
union UW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
{
    f: ManuallyDrop<T>,
}
union UW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
{
    f: ManuallyDrop<T>,
}

fn fi1<T: Iterator<Item: Copy, Item: Send>>() {}
fn fi2<T: Iterator<Item: Copy, Item: Copy>>() {}
fn fi3<T: Iterator<Item: 'static, Item: 'static>>() {}
fn fw1<T>()
where
    T: Iterator<Item: Copy, Item: Send>,
{
}
fn fw2<T>()
where
    T: Iterator<Item: Copy, Item: Copy>,
{
}
fn fw3<T>()
where
    T: Iterator<Item: 'static, Item: 'static>,
{
}

fn frpit1() -> impl Iterator<Item: Copy, Item: Send> {
    iter::empty::<u32>()
}
fn frpit2() -> impl Iterator<Item: Copy, Item: Copy> {
    iter::empty::<u32>()
}
fn frpit3() -> impl Iterator<Item: 'static, Item: 'static> {
    iter::empty::<u32>()
}
fn fapit1(_: impl Iterator<Item: Copy, Item: Send>) {}
fn fapit2(_: impl Iterator<Item: Copy, Item: Copy>) {}
fn fapit3(_: impl Iterator<Item: 'static, Item: 'static>) {}

type TAI1<T: Iterator<Item: Copy, Item: Send>> = T;
type TAI2<T: Iterator<Item: Copy, Item: Copy>> = T;
type TAI3<T: Iterator<Item: 'static, Item: 'static>> = T;
type TAW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
= T;
type TAW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
= T;
type TAW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
= T;

trait TRI1<T: Iterator<Item: Copy, Item: Send>> {}
trait TRI2<T: Iterator<Item: Copy, Item: Copy>> {}
trait TRI3<T: Iterator<Item: 'static, Item: 'static>> {}
trait TRS1: Iterator<Item: Copy, Item: Send> {}
trait TRS2: Iterator<Item: Copy, Item: Copy> {}
trait TRS3: Iterator<Item: 'static, Item: 'static> {}
trait TRW1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
{
}
trait TRW2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
{
}
trait TRW3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
{
}
trait TRSW1
where
    Self: Iterator<Item: Copy, Item: Send>,
{
}
trait TRSW2
where
    Self: Iterator<Item: Copy, Item: Copy>,
{
}
trait TRSW3
where
    Self: Iterator<Item: 'static, Item: 'static>,
{
}
trait TRA1 {
    type A: Iterator<Item: Copy, Item: Send>;
}
trait TRA2 {
    type A: Iterator<Item: Copy, Item: Copy>;
}
trait TRA3 {
    type A: Iterator<Item: 'static, Item: 'static>;
}

trait Trait {
    type Gat<T>;

    const ASSOC: i32;

    fn foo() -> impl Sized;
}

impl Trait for () {
    type Gat<T> = ();

    const ASSOC: i32 = 3;

    fn foo() {}
}

trait Subtrait: Trait<Gat<u32> = u32, Gat<u64> = u64> {}

fn f<T: Trait<Gat<i32> = (), Gat<i64> = ()>>() {
    let _: T::Gat<i32> = ();
    let _: T::Gat<i64> = ();
}

fn g<T: Trait<Gat<i32> = (), Gat<i64> = &'static str>>() {
    let _: T::Gat<i32> = ();
    let _: T::Gat<i64> = "";
}

fn uncallable(_: impl Iterator<Item = i32, Item = u32>) {}

fn callable(_: impl Iterator<Item = i32, Item = i32>) {}

fn uncallable_const(_: impl Trait<ASSOC = 3, ASSOC = 4>) {}

fn callable_const(_: impl Trait<ASSOC = 3, ASSOC = 3>) {}

fn uncallable_rtn(_: impl Trait<foo(..): Trait<ASSOC = 3>, foo(..): Trait<ASSOC = 4>>) {}

fn callable_rtn(_: impl Trait<foo(..): Send, foo(..): Send, foo(..): Eq>) {}

type Works = dyn Iterator<Item = i32, Item = i32>;
// ^~ ERROR conflicting associated type bounds

trait Trait2 {
    const ASSOC: u32;
}

type AlsoWorks = dyn Trait2<ASSOC = 3u32, ASSOC = 3u32>;
// ^~ ERROR conflicting associated type bounds

fn main() {
    callable(iter::empty::<i32>());
    callable_const(());
    callable_rtn(());
}
