# rust-semverver
[![Build Status](https://travis-ci.org/ibabushkin/rust-semverver.svg?branch=master)](https://travis-ci.org/ibabushkin/rust-semverver)

This repository is hosting a proof-of-concept implementation of an automatic tool checking
rust library crates for semantic versioning adherence. The goal is to provide an automated
command akin to `cargo clippy` that analyzes the current crate's souce code for changes
compared to the most recent version on `crates.io`.

## Background
The approach taken is to compile both versions of the crate to `rlib`s and to link them as
dependencies of a third crate. Then, a custom compiler driver is run on the resulting
crate and all necessary analysis is performed in that context.

The general concepts and aspects of the algorithms used are outlined in more detail in the
[proposal](https://summerofcode.withgoogle.com/projects/#5063973872336896).

## Installation
This is currently irrelevant, as the functionality is not yet implemented. Please check
back later.

## Usage
This is currently irrelevant, as the functionality is not yet implemented. Please check
back later.
