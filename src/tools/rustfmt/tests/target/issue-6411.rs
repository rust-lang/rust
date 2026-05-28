// rustfmt-edition: 2021

fn test_break() {
    'r#if: {
        break 'r#if;
    }

    'r#a: {
        break 'r#a;
    }
}

fn test_continue() {
    'r#if: {
        continue 'r#if;
    }

    'r#a: {
        continue 'r#a;
    }
}
