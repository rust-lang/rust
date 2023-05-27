use autodiff::autodiff;

#[autodiff(d_array, Reverse, Active, Duplicated)]
fn array(arr: &[[[f32; 2]; 2]; 2]) -> f32 {
    arr[0][0][0] * arr[1][1][1]
}

fn main() {
    let arr = [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]];
    let mut d_arr = [[[0.0; 2]; 2]; 2];

    d_array(&arr, &mut d_arr, 1.0);

    dbg!(&d_arr);
}

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
