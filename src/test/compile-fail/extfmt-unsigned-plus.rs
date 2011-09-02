// error-pattern:only valid in signed #fmt conversion

fn main() {
    // Can't use a sign on unsigned conversions
    #fmt["%+u", 10u];
}
