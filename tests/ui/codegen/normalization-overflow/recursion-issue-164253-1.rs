//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver=globally
//@ compile-flags: -Copt-level=0 -Arecursion_depth_exceeding_limit
//@ build-fail
//@ edition: 2021
//@ dont-check-compiler-stderr
//@ forbid-output: internal compiler error

use std::{
    fs::File,
    io::{Result, Write}};
enum Tree {
    Leaf(bool),
    Parent(Vec<Tree>)}
impl Tree {
    fn serialize<T: Write>(self, mut writer: T) -> Result<()> {
        match self {
            Tree::Leaf(_) => Ok(()),
            Tree::Parent(children) => {
                for child in children {
                    child.serialize(writer.by_ref())?}
        //~^ ERROR reached the recursion limit while instantiating
                Ok(())
            }
        }
    }
}
fn main() -> Result<()> {
    let buffer = File::create("")?;
    Tree::Parent(vec![]).serialize(buffer)
}
