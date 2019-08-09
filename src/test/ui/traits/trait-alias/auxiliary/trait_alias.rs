#![feature(trait_alias)]

pub trait SendSync = Send + Sync;
