// error-pattern:only valid in signed #fmt conversion

fn main() {
    // Can't use a space on unsigned conversions
    #ifmt["% u", 10u];
}
