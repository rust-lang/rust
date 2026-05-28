// ignore-tidy-tab

fn main() {
	bar; //~ ERROR cannot find value `bar`
}

fn foo() {
	"bar			boo" //~ ERROR mismatched types
}
