% The Guide to Rust Syntax

A *very* brief guide to Rust syntax. It assumes you are already familar with programming concepts. 

# Arguments

# Conditions

# `enum`

# Expressions

# Functions

Functions in Rust are introduced with the `fn` keyword, optional arguments are specified within parenthesis as comma separated `name: type` pairs, and `->` indicates the return type. You can ommit the return type for functions that do not return a value. Functions return the top level expression (note the return expression is not terminated with a semi colon).

~~~~
fn main() {
    let i = 7;

    fn increment(i:int) -> (int) {
       i + 1 
    }	

    let k = increment(i); // k=8
}
~~~~

# Loops

## `if`

## `loop`

## `while`

# Operators

# Patterns

# Return values

# Statements

# Structs and Tuples: `struct`

# Variables

There are two types of variable in Rust:

* `immutable` - the value cannot be changed. Introduced with `let mut`.
* `mutable` - the value of can be changed. Introduced with `let`.

~~~~
fn main() {

    let i = 7; // i Cannot be changed 

    let mut j = i +1; // j = 8
    
    j = 9; // j can be changed

}
~~~~
