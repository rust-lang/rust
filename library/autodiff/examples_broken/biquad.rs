use autodiff::autodiff;

#[derive(Debug)]
struct Biquad<const N: usize> {
    coeffs: [[f32; 5]; N],
}

impl<const N: usize> Biquad<N> {
    pub fn new() -> Self {
        Biquad { coeffs: [[0.0; 5]; N] }
    }

    pub fn process(&self, samples: &[f32], target: &[f32]) -> f32 {
        // do some horrible inefficient biquad filtering
        let mut samples = samples.to_vec();
        let mut samples_out = vec![0.0; samples.len()];

        for coeff_set in self.coeffs {
            for idx in 0..samples.len() {
                samples_out[idx] = coeff_set[0] * samples[idx];

                if idx > 0 {
                    samples_out[idx] += coeff_set[1] * samples[idx - 1] - 
                        coeff_set[3] * samples_out[idx - 1];
                }
                if idx > 1 {
                    samples_out[idx] += coeff_set[2] * samples[idx - 2] - 
                        coeff_set[4] * samples_out[idx - 2];
                }
            }

            (samples, samples_out) = (samples_out, samples);
        }

        samples_out.into_iter().zip(target.into_iter()).map(|(a, b)| a - b).sum()
    }

    #[autodiff(Self::process, Reverse, Active)]
    pub fn deriv(#[dup] &self, params: &mut Self, samples: &[f32], target: &[f32], ret_adj: f32);
}

fn main() {
    let biquad = Biquad::<10>::new();
    let mut dbiquad = Biquad::<10>::new();

    // create ramp and pulse train
    let signal = (0..1024).map(|x| (x as f32) / 1024.0).collect::<Vec<_>>();
    let target = (0..1024).map(|x| if x % 2 == 0 { 0.0 } else { 1.0 }).collect::<Vec<_>>();

    dbg!(&biquad.process(&signal, &target));
    biquad.deriv(&mut dbiquad, &signal, &target, 1.0);

    dbg!(&dbiquad);
}
