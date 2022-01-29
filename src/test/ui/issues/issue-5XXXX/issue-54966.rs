// issue-54966: ICE returning an unknown type with impl FnMut

fn generate_duration() -> Oper<impl FnMut()> {}
//~^ ERROR cannot find type `Oper` in this scope

fn main() {}
