struct Project;
struct Value;

static settings_dir: String = format!("");
//~^ ERROR cannot call non-const fn
//~| ERROR is not yet stable as a const

fn from_string(_: String) -> Value {
    Value
}
fn set_editor(_: Value) {}

fn main() {
    let settings_data = from_string(settings_dir);
    //~^ ERROR cannot move out of static item
    let args: i32 = 0;

    match args {
        ref x if x == &0 => set_editor(settings_data),
        ref x if x == &1 => set_editor(settings_data),
        _ => unimplemented!(),
    }
}
