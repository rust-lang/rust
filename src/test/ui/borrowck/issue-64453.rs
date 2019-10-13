struct Project;
struct Value;

static settings_dir: String = format!("");
//~^ ERROR [E0019]
//~| ERROR [E0015]
//~| ERROR [E0015]

fn from_string(_: String) -> Value {
    Value
}
fn set_editor(_: Value) {}

fn main() {
    let settings_data = from_string(settings_dir);
    //~^ ERROR cannot move out of static item `settings_dir` [E0507]
    let args: i32 = 0;

    match args {
        ref x if x == &0 => set_editor(settings_data),
        ref x if x == &1 => set_editor(settings_data),
        _ => unimplemented!(),
    }
}
