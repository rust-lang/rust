use autodiff::autodiff;

#[autodiff(d_sqrt, Reverse, Active)]
fn sqrt(#[active] a: f32, #[dup] b: &f32, c: &f32, #[active] d: f32) -> f32 {
    a * (b * b + c*c*d*d).sqrt()
}

fn main() {
    let mut d_b = 0.0;

    let (d_a, d_d) = d_sqrt(1.0, &1.0, &mut d_b, &1.0, 1.0, 1.0);
    dbg!(d_a, d_b, d_d);
}

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
