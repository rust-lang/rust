//@ edition: 2024
//@ run-pass

#![feature(associated_const_equality, return_type_notation)]
#![allow(dead_code, refining_impl_trait_internal, type_alias_bounds)]

use std::iter;
use std::mem::ManuallyDrop;

struct Si1<T: Iterator<Item: Copy, Item: Send>> {
    f: T,
}
struct Si2<T: Iterator<Item: Copy, Item: Copy>> {
    f: T,
}
struct Si3<T: Iterator<Item: 'static, Item: 'static>> {
    f: T,
}
struct Sw1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
{
    f: T,
}
struct Sw2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
{
    f: T,
}
struct Sw3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
{
    f: T,
}

enum Ei1<T: Iterator<Item: Copy, Item: Send>> {
    V(T),
}
enum Ei2<T: Iterator<Item: Copy, Item: Copy>> {
    V(T),
}
enum Ei3<T: Iterator<Item: 'static, Item: 'static>> {
    V(T),
}
enum Ew1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
{
    V(T),
}
enum Ew2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
{
    V(T),
}
enum Ew3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
{
    V(T),
}

union Ui1<T: Iterator<Item: Copy, Item: Send>> {
    f: ManuallyDrop<T>,
}
union Ui2<T: Iterator<Item: Copy, Item: Copy>> {
    f: ManuallyDrop<T>,
}
union Ui3<T: Iterator<Item: 'static, Item: 'static>> {
    f: ManuallyDrop<T>,
}
union Uw1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
{
    f: ManuallyDrop<T>,
}
union Uw2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
{
    f: ManuallyDrop<T>,
}
union Uw3<T>
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

fn rpit1() -> impl Iterator<Item: Copy, Item: Send> {
    iter::empty::<u32>()
}
fn rpit2() -> impl Iterator<Item: Copy, Item: Copy> {
    iter::empty::<u32>()
}
fn rpit3() -> impl Iterator<Item: 'static, Item: 'static> {
    iter::empty::<u32>()
}
fn apit1(_: impl Iterator<Item: Copy, Item: Send>) {}
fn apit2(_: impl Iterator<Item: Copy, Item: Copy>) {}
fn apit3(_: impl Iterator<Item: 'static, Item: 'static>) {}

type Tait1<T: Iterator<Item: Copy, Item: Send>> = T;
type Tait2<T: Iterator<Item: Copy, Item: Copy>> = T;
type Tait3<T: Iterator<Item: 'static, Item: 'static>> = T;
type Taw1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
= T;
type Taw2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
= T;
type Taw3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
= T;

trait Tri1<T: Iterator<Item: Copy, Item: Send>> {}
trait Tri2<T: Iterator<Item: Copy, Item: Copy>> {}
trait Tri3<T: Iterator<Item: 'static, Item: 'static>> {}
trait Trs1: Iterator<Item: Copy, Item: Send> {}
trait Trs2: Iterator<Item: Copy, Item: Copy> {}
trait Trs3: Iterator<Item: 'static, Item: 'static> {}
trait Trw1<T>
where
    T: Iterator<Item: Copy, Item: Send>,
{
}
trait Trw2<T>
where
    T: Iterator<Item: Copy, Item: Copy>,
{
}
trait Trw3<T>
where
    T: Iterator<Item: 'static, Item: 'static>,
{
}
trait Trsw1
where
    Self: Iterator<Item: Copy, Item: Send>,
{
}
trait Trsw2
where
    Self: Iterator<Item: Copy, Item: Copy>,
{
}
trait Trsw3
where
    Self: Iterator<Item: 'static, Item: 'static>,
{
}
trait Tra1 {
    type A: Iterator<Item: Copy, Item: Send>;
}
trait Tra2 {
    type A: Iterator<Item: Copy, Item: Copy>;
}
trait Tra3 {
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

trait Trait2 {
    const ASSOC: u32;
}

trait Trait3 {
    fn foo() -> impl Iterator<Item = i32, Item = u32>;
}

fn main() {
    callable(iter::empty::<i32>());
    callable_const(());
    callable_rtn(());
}
