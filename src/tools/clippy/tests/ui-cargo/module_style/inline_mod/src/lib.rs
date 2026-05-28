#![warn(clippy::inline_modules)]

pub mod other;
#[path = "qux/mod.rs"]
pub mod something;
#[path = "foo.rs"]
pub mod stuff;

pub mod test_nested_inline_mods {
    mod bar {
        mod baz {}
    }
}

#[cfg(test)]
mod escaped_test_mod {
    mod bar {}
}

mod partially_escaped_test_mod {
    #[cfg(test)]
    mod tests {
        mod bar {}
    }
    mod baz {}
}

macro_rules! inline_mod_from_expansion {
    () => {
        mod _foo {}
    };
}

inline_mod_from_expansion!();
