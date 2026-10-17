// rustfmt-edition: 2021

// Simple idempotence test for raw lifetimes.

fn test<'r#gen>() -> &'r#gen () {
    // Test raw lifetimes...
}

fn label() {
    'r#label: {
        // Test raw labels.
    }
}

fn main() {}
