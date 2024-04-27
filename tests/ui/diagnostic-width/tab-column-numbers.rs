// Test for #109537: ensure that column numbers are correctly generated when using hard tabs.
//@ aux-build:tab_column_numbers.rs

// ignore-tidy-tab

extern crate tab_column_numbers;

fn main() {
	let s = tab_column_numbers::S;
	s.method();
	//~^ ERROR method `method` is private
}
