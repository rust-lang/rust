//@ add-minicore
//@ needs-llvm-components: nvptx
//@ compile-flags: --target nvptx64-nvidia-cuda --emit link
//@ ignore-backends: gcc
#![crate_type = "rlib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

struct Foo(&'static Foo);
impl Sync for Foo {}

static A: Foo = Foo(&A); //~ ERROR static initializer forms a cycle involving `A`

static B0: Foo = Foo(&B1); //~ ERROR static initializer forms a cycle involving `B0`
static B1: Foo = Foo(&B0);

static C0: Foo = Foo(&C1); //~ ERROR static initializer forms a cycle involving `C0`
static C1: Foo = Foo(&C2);
static C2: Foo = Foo(&C0);

struct Bar(&'static u32);
impl Sync for Bar {}

static BAR: Bar = Bar(&INT);
static INT: u32 = 42u32;
