// Make sure suggestion for removal of a span that covers multiple lines is properly highlighted.
//@ compile-flags: --error-format=human --color=always
//@ edition:2018
//@ only-linux
// ignore-tidy-tab
// We use `\t` instead of spaces for indentation to ensure that the highlighting logic properly
// accounts for replaced characters (like we do for `\t` with `    `). The naïve way of highlighting
// could be counting chars of the original code, instead of operating on the code as it is being
// displayed.
use std::collections::{HashMap, HashSet};
fn foo() -> Vec<(bool, HashSet<u8>)> {
	let mut hm = HashMap::<bool, Vec<HashSet<u8>>>::new();
	hm.into_iter()
		.map(|(is_true, ts)| {
			ts.into_iter()
				.map(|t| {
					(
						is_true,
						t,
					)
				}).flatten()
		})
		.flatten()
		.collect()
}
fn bar() -> Vec<(bool, HashSet<u8>)> {
	let mut hm = HashMap::<bool, Vec<HashSet<u8>>>::new();
	hm.into_iter()
		.map(|(is_true, ts)| {
			ts.into_iter()
				.map(|t| (is_true, t))
				.flatten()
		})
		.flatten()
		.collect()
}
fn baz() -> Vec<(bool, HashSet<u8>)> {
	let mut hm = HashMap::<bool, Vec<HashSet<u8>>>::new();
	hm.into_iter()
		.map(|(is_true, ts)| {
			ts.into_iter().map(|t| {
				(is_true, t)
			}).flatten()
		})
		.flatten()
		.collect()
}
fn bay() -> Vec<(bool, HashSet<u8>)> {
	let mut hm = HashMap::<bool, Vec<HashSet<u8>>>::new();
	hm.into_iter()
		.map(|(is_true, ts)| {
			ts.into_iter()
				.map(|t| (is_true, t)).flatten()
		})
		.flatten()
		.collect()
}
fn main() {}
