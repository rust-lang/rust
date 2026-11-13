trait A<T=Self> {}

fn together_we_will_rule_the_galaxy(son: &dyn A) {}
//~^ ERROR E0393

fn main() {
}
