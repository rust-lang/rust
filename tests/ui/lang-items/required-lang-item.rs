//@ edition: 2018

#![feature(lang_items, no_core)]
#![no_core]
#![no_main]

#[lang="copy"] pub trait Copy { }
#[lang="sized"] pub trait Sized { }

async fn x() {} //~ ERROR requires `ResumeTy` lang_item
