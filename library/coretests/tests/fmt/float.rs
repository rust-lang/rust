use core::fmt::{self, Write};

#[test]
fn test_format_f64() {
    assert_eq!("1", format!("{:.0}", 1.0f64));
    assert_eq!("9", format!("{:.0}", 9.4f64));
    assert_eq!("10", format!("{:.0}", 9.9f64));
    assert_eq!("9.8", format!("{:.1}", 9.849f64));
    assert_eq!("9.9", format!("{:.1}", 9.851f64));
    assert_eq!("0", format!("{:.0}", 0.5f64));
    assert_eq!("1.23456789e6", format!("{:e}", 1234567.89f64));
    assert_eq!("1.23456789e3", format!("{:e}", 1234.56789f64));
    assert_eq!("1.23456789E6", format!("{:E}", 1234567.89f64));
    assert_eq!("1.23456789E3", format!("{:E}", 1234.56789f64));
    assert_eq!("0.0", format!("{:?}", 0.0f64));
    assert_eq!("1.01", format!("{:?}", 1.01f64));

    let high_cutoff = 1e16_f64;
    assert_eq!("1e16", format!("{:?}", high_cutoff));
    assert_eq!("-1e16", format!("{:?}", -high_cutoff));
    assert!(!is_exponential(&format!("{:?}", high_cutoff * (1.0 - 2.0 * f64::EPSILON))));
    assert_eq!("-3.0", format!("{:?}", -3f64));
    assert_eq!("0.0001", format!("{:?}", 0.0001f64));
    assert_eq!("9e-5", format!("{:?}", 0.00009f64));
    assert_eq!("1234567.9", format!("{:.1?}", 1234567.89f64));
    assert_eq!("1234.6", format!("{:.1?}", 1234.56789f64));
}

#[test]
fn test_format_f64_rounds_ties_to_even() {
    // Use only values exactly representable in IEEE floating-point, i.e.,
    // multiples of 1/2^n.
    assert_eq!("0", format!("{:.0}", 0.5f64));
    assert_eq!("2", format!("{:.0}", 1.5f64));
    assert_eq!("2", format!("{:.0}", 2.5f64));
    assert_eq!("4", format!("{:.0}", 3.5f64));
    assert_eq!("4", format!("{:.0}", 4.5f64));
    assert_eq!("6", format!("{:.0}", 5.5f64));
    assert_eq!("128", format!("{:.0}", 127.5f64));
    assert_eq!("128", format!("{:.0}", 128.5f64));
    assert_eq!("0.2", format!("{:.1}", 0.25f64));
    assert_eq!("0.8", format!("{:.1}", 0.75f64));
    assert_eq!("0.12", format!("{:.2}", 0.125f64));
    assert_eq!("0.88", format!("{:.2}", 0.875f64));
    assert_eq!("0.062", format!("{:.3}", 0.0625f64));
    assert_eq!("-0", format!("{:.0}", -0.5f64));
    assert_eq!("-2", format!("{:.0}", -1.5f64));
    assert_eq!("-2", format!("{:.0}", -2.5f64));
    assert_eq!("-4", format!("{:.0}", -3.5f64));
    assert_eq!("-4", format!("{:.0}", -4.5f64));
    assert_eq!("-6", format!("{:.0}", -5.5f64));
    assert_eq!("-128", format!("{:.0}", -127.5f64));
    assert_eq!("-128", format!("{:.0}", -128.5f64));
    assert_eq!("-0.2", format!("{:.1}", -0.25f64));
    assert_eq!("-0.8", format!("{:.1}", -0.75f64));
    assert_eq!("-0.12", format!("{:.2}", -0.125f64));
    assert_eq!("-0.88", format!("{:.2}", -0.875f64));
    assert_eq!("-0.062", format!("{:.3}", -0.0625f64));

    assert_eq!("2e0", format!("{:.0e}", 1.5f64));
    assert_eq!("2e0", format!("{:.0e}", 2.5f64));
    assert_eq!("4e0", format!("{:.0e}", 3.5f64));
    assert_eq!("4e0", format!("{:.0e}", 4.5f64));
    assert_eq!("6e0", format!("{:.0e}", 5.5f64));
    assert_eq!("1.28e2", format!("{:.2e}", 127.5f64));
    assert_eq!("1.28e2", format!("{:.2e}", 128.5f64));
    assert_eq!("-2e0", format!("{:.0e}", -1.5f64));
    assert_eq!("-2e0", format!("{:.0e}", -2.5f64));
    assert_eq!("-4e0", format!("{:.0e}", -3.5f64));
    assert_eq!("-4e0", format!("{:.0e}", -4.5f64));
    assert_eq!("-6e0", format!("{:.0e}", -5.5f64));
    assert_eq!("-1.28e2", format!("{:.2e}", -127.5f64));
    assert_eq!("-1.28e2", format!("{:.2e}", -128.5f64));

    assert_eq!("2E0", format!("{:.0E}", 1.5f64));
    assert_eq!("2E0", format!("{:.0E}", 2.5f64));
    assert_eq!("4E0", format!("{:.0E}", 3.5f64));
    assert_eq!("4E0", format!("{:.0E}", 4.5f64));
    assert_eq!("6E0", format!("{:.0E}", 5.5f64));
    assert_eq!("1.28E2", format!("{:.2E}", 127.5f64));
    assert_eq!("1.28E2", format!("{:.2E}", 128.5f64));
    assert_eq!("-2E0", format!("{:.0E}", -1.5f64));
    assert_eq!("-2E0", format!("{:.0E}", -2.5f64));
    assert_eq!("-4E0", format!("{:.0E}", -3.5f64));
    assert_eq!("-4E0", format!("{:.0E}", -4.5f64));
    assert_eq!("-6E0", format!("{:.0E}", -5.5f64));
    assert_eq!("-1.28E2", format!("{:.2E}", -127.5f64));
    assert_eq!("-1.28E2", format!("{:.2E}", -128.5f64));
}

#[test]
fn test_format_f64_round_leading_digit() {
    // Only 0.5 is exactly representable in IEEE floating-point,
    assert_eq!("0.1", format!("{:.1}", 5e-2f64));
    assert_eq!("0.01", format!("{:.2}", 5e-3f64));
    assert_eq!("0.001", format!("{:.3}", 5e-4f64));
    assert_eq!("0.0001", format!("{:.4}", 5e-5f64));
    assert_eq!("0.00001", format!("{:.5}", 5e-6f64));
    assert_eq!("0.000000", format!("{:.6}", 5e-7f64));
    assert_eq!("0.0000000", format!("{:.7}", 5e-8f64));
    assert_eq!("0.00000001", format!("{:.8}", 5e-9f64));
    assert_eq!("0.000000001", format!("{:.9}", 5e-10f64));
    assert_eq!("0.0000000001", format!("{:.10}", 5e-11f64));
    assert_eq!("0.00000000000", format!("{:.11}", 5e-12f64));
    assert_eq!("0.000000000000", format!("{:.12}", 5e-13f64));
    assert_eq!("0.0000000000001", format!("{:.13}", 5e-14f64));
    assert_eq!("0.00000000000000", format!("{:.14}", 5e-15f64));
    assert_eq!("0.000000000000001", format!("{:.15}", 5e-16f64));
}

#[test]
fn test_format_f32() {
    assert_eq!("1", format!("{:.0}", 1.0f32));
    assert_eq!("9", format!("{:.0}", 9.4f32));
    assert_eq!("10", format!("{:.0}", 9.9f32));
    assert_eq!("9.8", format!("{:.1}", 9.849f32));
    assert_eq!("9.9", format!("{:.1}", 9.851f32));
    assert_eq!("0", format!("{:.0}", 0.5f32));
    assert_eq!("1.2345679e6", format!("{:e}", 1234567.89f32));
    assert_eq!("1.2345679e3", format!("{:e}", 1234.56789f32));
    assert_eq!("1.2345679E6", format!("{:E}", 1234567.89f32));
    assert_eq!("1.2345679E3", format!("{:E}", 1234.56789f32));
    assert_eq!("0.0", format!("{:?}", 0.0f32));
    assert_eq!("1.01", format!("{:?}", 1.01f32));

    let high_cutoff = 1e16_f32;
    assert_eq!("1e16", format!("{:?}", high_cutoff));
    assert_eq!("-1e16", format!("{:?}", -high_cutoff));
    assert!(!is_exponential(&format!("{:?}", high_cutoff * (1.0 - 2.0 * f32::EPSILON))));
    assert_eq!("-3.0", format!("{:?}", -3f32));
    assert_eq!("0.0001", format!("{:?}", 0.0001f32));
    assert_eq!("9e-5", format!("{:?}", 0.00009f32));
    assert_eq!("1234567.9", format!("{:.1?}", 1234567.89f32));
    assert_eq!("1234.6", format!("{:.1?}", 1234.56789f32));
}

#[test]
fn test_format_f32_rounds_ties_to_even() {
    // Use only values exactly representable in IEEE floating-point, i.e.,
    // multiples of 1/2^n.
    assert_eq!("0", format!("{:.0}", 0.5f32));
    assert_eq!("2", format!("{:.0}", 1.5f32));
    assert_eq!("2", format!("{:.0}", 2.5f32));
    assert_eq!("4", format!("{:.0}", 3.5f32));
    assert_eq!("4", format!("{:.0}", 4.5f32));
    assert_eq!("6", format!("{:.0}", 5.5f32));
    assert_eq!("128", format!("{:.0}", 127.5f32));
    assert_eq!("128", format!("{:.0}", 128.5f32));
    assert_eq!("0.2", format!("{:.1}", 0.25f32));
    assert_eq!("0.8", format!("{:.1}", 0.75f32));
    assert_eq!("0.12", format!("{:.2}", 0.125f32));
    assert_eq!("0.88", format!("{:.2}", 0.875f32));
    assert_eq!("0.062", format!("{:.3}", 0.0625f32));
    assert_eq!("-0", format!("{:.0}", -0.5f32));
    assert_eq!("-2", format!("{:.0}", -1.5f32));
    assert_eq!("-2", format!("{:.0}", -2.5f32));
    assert_eq!("-4", format!("{:.0}", -3.5f32));
    assert_eq!("-4", format!("{:.0}", -4.5f32));
    assert_eq!("-6", format!("{:.0}", -5.5f32));
    assert_eq!("-128", format!("{:.0}", -127.5f32));
    assert_eq!("-128", format!("{:.0}", -128.5f32));
    assert_eq!("-0.2", format!("{:.1}", -0.25f32));
    assert_eq!("-0.8", format!("{:.1}", -0.75f32));
    assert_eq!("-0.12", format!("{:.2}", -0.125f32));
    assert_eq!("-0.88", format!("{:.2}", -0.875f32));
    assert_eq!("-0.062", format!("{:.3}", -0.0625f32));

    assert_eq!("2e0", format!("{:.0e}", 1.5f32));
    assert_eq!("2e0", format!("{:.0e}", 2.5f32));
    assert_eq!("4e0", format!("{:.0e}", 3.5f32));
    assert_eq!("4e0", format!("{:.0e}", 4.5f32));
    assert_eq!("6e0", format!("{:.0e}", 5.5f32));
    assert_eq!("1.28e2", format!("{:.2e}", 127.5f32));
    assert_eq!("1.28e2", format!("{:.2e}", 128.5f32));
    assert_eq!("-2e0", format!("{:.0e}", -1.5f32));
    assert_eq!("-2e0", format!("{:.0e}", -2.5f32));
    assert_eq!("-4e0", format!("{:.0e}", -3.5f32));
    assert_eq!("-4e0", format!("{:.0e}", -4.5f32));
    assert_eq!("-6e0", format!("{:.0e}", -5.5f32));
    assert_eq!("-1.28e2", format!("{:.2e}", -127.5f32));
    assert_eq!("-1.28e2", format!("{:.2e}", -128.5f32));

    assert_eq!("2E0", format!("{:.0E}", 1.5f32));
    assert_eq!("2E0", format!("{:.0E}", 2.5f32));
    assert_eq!("4E0", format!("{:.0E}", 3.5f32));
    assert_eq!("4E0", format!("{:.0E}", 4.5f32));
    assert_eq!("6E0", format!("{:.0E}", 5.5f32));
    assert_eq!("1.28E2", format!("{:.2E}", 127.5f32));
    assert_eq!("1.28E2", format!("{:.2E}", 128.5f32));
    assert_eq!("-2E0", format!("{:.0E}", -1.5f32));
    assert_eq!("-2E0", format!("{:.0E}", -2.5f32));
    assert_eq!("-4E0", format!("{:.0E}", -3.5f32));
    assert_eq!("-4E0", format!("{:.0E}", -4.5f32));
    assert_eq!("-6E0", format!("{:.0E}", -5.5f32));
    assert_eq!("-1.28E2", format!("{:.2E}", -127.5f32));
    assert_eq!("-1.28E2", format!("{:.2E}", -128.5f32));
}

#[test]
fn test_format_f64_max_precision_exponential() {
    struct ExactExpWriter {
        prefix: &'static [u8],
        zeroes_remaining: u32,
        suffix: &'static [u8],
        prefix_pos: usize,
        suffix_pos: usize,
        total_len: u32,
    }

    impl ExactExpWriter {
        fn new(prefix: &'static str, suffix: &'static str) -> Self {
            Self {
                prefix: prefix.as_bytes(),
                zeroes_remaining: u16::MAX.into(),
                suffix: suffix.as_bytes(),
                prefix_pos: 0,
                suffix_pos: 0,
                total_len: 0,
            }
        }

        fn finish(self) {
            assert_eq!(self.prefix_pos, self.prefix.len());
            assert_eq!(self.zeroes_remaining, 0);
            assert_eq!(self.suffix_pos, self.suffix.len());
            assert_eq!(self.total_len, u32::from(u16::MAX) + 4);
        }
    }

    impl Write for ExactExpWriter {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            for byte in s.bytes() {
                self.total_len += 1;

                if self.prefix_pos < self.prefix.len() {
                    assert_eq!(byte, self.prefix[self.prefix_pos]);
                    self.prefix_pos += 1;
                } else if self.zeroes_remaining > 0 {
                    assert_eq!(byte, b'0');
                    self.zeroes_remaining -= 1;
                } else {
                    assert!(self.suffix_pos < self.suffix.len());
                    assert_eq!(byte, self.suffix[self.suffix_pos]);
                    self.suffix_pos += 1;
                }
            }

            Ok(())
        }
    }

    fn assert_exact_exp(args: fmt::Arguments<'_>, prefix: &'static str, suffix: &'static str) {
        let mut writer = ExactExpWriter::new(prefix, suffix);
        fmt::write(&mut writer, args).unwrap();
        writer.finish();
    }

    assert_exact_exp(format_args!("{:.65535e}", 0.0f64), "0.", "e0");
    assert_exact_exp(format_args!("{:.65535e}", 1.0f64), "1.", "e0");
    assert_exact_exp(format_args!("{:.65535E}", 0.0f64), "0.", "E0");
    assert_exact_exp(format_args!("{:.65535E}", 1.0f64), "1.", "E0");
    assert_exact_exp(format_args!("{:65535.65535e}", 1.0f64), "1.", "e0");
}

fn is_exponential(s: &str) -> bool {
    s.contains("e") || s.contains("E")
}
