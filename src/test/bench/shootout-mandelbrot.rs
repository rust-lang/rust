use std::cast::transmute;
use std::from_str::FromStr;
use std::i32::range;
use std::libc::{STDOUT_FILENO, c_int, fdopen, fputc};
use std::os;

static ITER: uint = 50;
static LIMIT: f64 = 2.0;

#[fixed_stack_segment]
fn main() {
    unsafe {
        let w: i32 = FromStr::from_str(os::args()[1]).get();
        let h = w;
        let mut byte_acc: i8 = 0;
        let mut bit_num: i32 = 0;

        println(fmt!("P4\n%d %d", w as int, h as int));

        let mode = "w";
        let stdout = fdopen(STDOUT_FILENO as c_int, transmute(&mode[0]));

        for range(0, h) |y| {
            let y = y as f64;
            for range(0, w) |x| {
                let mut (Zr, Zi, Tr, Ti) = (0f64, 0f64, 0f64, 0f64);
                let Cr = 2.0 * (x as f64) / (w as f64) - 1.5;
                let Ci = 2.0 * (y as f64) / (h as f64) - 1.0;

                for ITER.times {
                    if Tr + Ti > LIMIT * LIMIT {
                        break;
                    }

                    Zi = 2.0*Zr*Zi + Ci;
                    Zr = Tr - Ti + Cr;
                    Tr = Zr * Zr;
                    Ti = Zi * Zi;
                }

                byte_acc <<= 1;
                if Tr + Ti <= LIMIT * LIMIT {
                    byte_acc |= 1;
                }

                bit_num += 1;

                if bit_num == 8 {
                    fputc(byte_acc as c_int, stdout);
                    byte_acc = 0;
                    bit_num = 0;
                } else if x == w - 1 {
                    byte_acc <<= 8 - w%8;
                    fputc(byte_acc as c_int, stdout);
                    byte_acc = 0;
                    bit_num = 0;
                }
            }
        }
    }
}
