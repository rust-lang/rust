// rustfmt-reorder_imports: true

// Ensure that a use at the start of an inline module is correctly formatted.
mod foo {
    use bar;
}

// Ensure that an indented `use` gets the correct indentation.
mod foo {
    use bar;
}
