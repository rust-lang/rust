struct Project;
struct Value;

static settings_dir: String = format!("");
//~^ ERROR `match` is not allowed in a `static`

fn from_string(_: String) -> Value {
    Value
}
fn set_editor(_: Value) {}

fn main() {
    let settings_data = from_string(settings_dir);
    let args: i32 = 0;

    match args {
        ref x if x == &0 => set_editor(settings_data),
        ref x if x == &1 => set_editor(settings_data),
        _ => unimplemented!(),
    }
}
