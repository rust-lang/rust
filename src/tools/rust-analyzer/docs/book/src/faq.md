# Troubleshooting FAQ

### I see a warning "Variable `None` should have snake_case name, e.g. `none`"

rust-analyzer fails to resolve `None`, and thinks you are binding to a variable
named `None`. That's usually a sign of a corrupted sysroot. Try removing and re-installing
it: `rustup component remove rust-src` then `rustup component install rust-src`.
