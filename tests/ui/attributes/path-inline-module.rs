//@ check-pass

fn main() {}

#[path = "foo.rs"] //~ WARN unused attribute [unused_attributes]
mod inline_module {}

mod inline_module_with_inner_path {
    #![path = "foo.rs"] //~ WARN unused attribute [unused_attributes]
}

#[path = "auxiliary/foo.rs"]
mod outline_module; // Should not warn

mod inline_with_inner_path {
    #![path = "auxiliary"]

    #[path = "foo.rs"] // Should not warn
    mod file_submodule;
}

mod inline_with_inline_child {
    #![path = "auxiliary"]

    #[path = "foo.rs"] //~ WARN unused attribute [unused_attributes]
    mod nested_inline_module {}
}

#[path = "auxiliary"]
mod inline_parent_with_file_child {
    #[path = "foo.rs"] // Should not warn
    mod file_submodule;
}

#[path = "auxiliary"]
mod inline_parent_with_inline_child {
    #[path = "foo.rs"] //~ WARN unused attribute [unused_attributes]
    mod inline_submodule {}
}

#[path = "auxiliary"]
mod inline_parent_with_nested_file_child {
    mod sub {
        #[path = "foo.rs"] // Should not warn
        mod file_submodule;
    }
}

mod inline_with_inner_path_and_nested_file_child {
    #![path = "auxiliary"]

    #[path = "sub"]
    mod middle {
        #[path = "foo.rs"] // Should not warn
        mod file_submodule;
    }
}

#[path = "auxiliary"] //~ WARN unused attribute [unused_attributes]
mod inline_parent_with_nested_inline_child {
    mod middle {
        mod inline_submodule {}
    }
}

#[path = "auxiliary"]
mod inline_parent_with_nested_inline_path {

    #[path = ""]
    mod middle {
        #[path = "foo.rs"] //~ WARN unused attribute [unused_attributes]
        mod inline_submodule {}
    }
}
