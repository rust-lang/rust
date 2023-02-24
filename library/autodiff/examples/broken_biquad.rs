#![feature(bench_black_box)]
use autodiff::autodiff;

struct Biquad<const N: usize> {
    coeffs: [[f32; 5]; N],
}

impl<const N: usize> Biquad<N> {
    pub fn new() -> Self {
        Biquad {
            coeffs: [[0.0; 5]; N],
        }
    }

    pub fn process(&self, mut samples: &[f32], target: &[f32]) -> f32 {
        let mut samples = samples.to_vec();
        for coeff_set in self.coeffs {
            samples = samples.windows(5).map(|x| 
                (coeff_set[0] * x[0] + coeff_set[1] * x[1] + coeff_set[2] * x[2]) / (1.0 + coeff_set[3] * x[3] + coeff_set[4] * x[4])).collect();
        }

        samples.into_iter().zip(target.into_iter()).map(|(a, b)| a-b).sum()
    }

    #[autodiff_into]
    fn _diff_deriv(&self, samples: &[f32], target: &[f32]) -> f32 {
        Self::process(self, samples, target)
    }
    #[autodiff_into(Reverse, Active, Duplicated, Const, Const)]
    fn deriv(&self, params: &mut Self, samples: &[f32], target: &[f32], ret_adj: f32) {
        std::hint::black_box((
            Self::_diff_deriv(self, samples, target),
            &params,
            &self,
            &samples,
            &target,
            &ret_adj,
        ));
    }

    //#[autodiff(Self::process, Reverse, Active)]
    //pub fn deriv(
    //    #[dup] &self, params: &mut Self,
    //    samples: &[f32], target: &[f32], ret_adj: f32);
}

fn main() {
    let biquad = Biquad::<10>::new();
    let mut dbiquad = Biquad::<20>::new();
    let signal = vec![0.0; 1024];
    let target = vec![0.0; 1024];

    biquad.process(&signal, &target);
    biquad.deriv(&mut dbiquad, &signal, &target, 1.0);
}

fn main() {
    let biquad1 = Biquad::<10>::new();
    let biquad2 = Biquad::<20>::new();

    assert!(size_of(biquad1) != size_of(biquad2))
}
