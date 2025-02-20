#![feature(const_trait_impl, sized_hierarchy)]

use std::marker::{MetaSized, PointeeSized};

pub trait ConstSizedTr {}

impl<T: const Sized> ConstSizedTr for T {}

pub trait SizedTr {}

impl<T: Sized> SizedTr for T {}

pub trait NegSizedTr {}

impl<T: ?Sized> NegSizedTr for T {}

pub trait ConstMetaSizedTr {}

impl<T: const MetaSized> ConstMetaSizedTr for T {}

pub trait MetaSizedTr {}

impl<T: MetaSized> MetaSizedTr for T {}

pub trait PointeeSizedTr: PointeeSized {}

impl<T: PointeeSized> PointeeSizedTr for T {}
