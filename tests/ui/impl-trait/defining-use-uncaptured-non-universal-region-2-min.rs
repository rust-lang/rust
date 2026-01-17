trait Trait {}
impl Trait for () {}
fn build(
    mut commands: impl Trait,
) -> Option<impl Trait> {
    // Different combinations result in the following equalities (with different orderings)
    // () == build::<build::<I>::opaque>::opaque == build::<I>::opaque>
    match () {
        //_ => Some(()),
        _ => build(commands),
        _ => {
            let further_commands = match build(commands) {
                Some(c) => c,
                None => return None,
            };
            build(further_commands)
        }
    }
}

fn main() {}
