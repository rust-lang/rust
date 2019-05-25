fn main() {}

trait T {
    fn qux() -> Option<usize> {
        let _ = if true {
        });
//~^ ERROR expected one of `async`, `const`, `extern`, `fn`, `type`, `unsafe`, or `}`, found `;`
//~^^ ERROR expected one of `.`, `;`, `?`, `else`, or an operator, found `}`
//~^^^ ERROR 6:11: 6:12: expected identifier, found `;`
//~^^^^ ERROR missing `fn`, `type`, or `const` for trait-item declaration
        Some(4)
    }
