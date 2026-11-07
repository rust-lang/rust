#![feature(trait_alias)]

pub trait ExtAlias0 = Copy + Iterator<Item = u8>;

pub trait ExtAlias1<'a, T: 'a + Clone, const N: usize> = From<[&'a T; N]>;

pub trait ExtAlias2<T> = where T: From<String>, String: Into<T>;

pub trait ExtAlias3 = Sized;

pub trait ExtAlias4 = where Self: Sized;

pub trait ExtAlias5 = ;
