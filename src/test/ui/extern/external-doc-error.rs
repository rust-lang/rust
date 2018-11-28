// normalize-stderr-test: "not-a-file.md:.*\(" -> "not-a-file.md: $$FILE_NOT_FOUND_MSG ("

#![feature(external_doc)]

#[doc(include = "not-a-file.md")] //~ ERROR: couldn't read
pub struct SomeStruct;

#[doc(include)]
pub struct MissingPath; //~^ ERROR expected path
                        //~| HELP provide a file path with `=`
                        //~| SUGGESTION include = "<path>"

#[doc(include("../README.md"))]
pub struct InvalidPathSyntax; //~^ ERROR expected path
                              //~| HELP provide a file path with `=`
                              //~| SUGGESTION include = "../README.md"

#[doc(include = 123)]
pub struct InvalidPathType; //~^ ERROR expected path
                            //~| HELP provide a file path with `=`
                            //~| SUGGESTION include = "<path>"

#[doc(include(123))]
pub struct InvalidPathSyntaxAndType; //~^ ERROR expected path
                                     //~| HELP provide a file path with `=`
                                     //~| SUGGESTION include = "<path>"

fn main() {}
