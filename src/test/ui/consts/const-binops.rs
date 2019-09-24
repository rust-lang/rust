// run-pass

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6,
                "{} is not approximately equal to {}", *a, *b);
    })
}

static A: isize = -4 + 3;
static A2: usize = 3 + 3;
static B: f64 = 3.0 + 2.7;

static C: isize = 3 - 4;
static D: usize = 3 - 3;
static E: f64 = 3.0 - 2.7;

static E2: isize = -3 * 3;
static F: usize = 3 * 3;
static G: f64 = 3.3 * 3.3;

static H: isize = 3 / -1;
static I: usize = 3 / 3;
static J: f64 = 3.3 / 3.3;

static N: bool = true && false;

static O: bool = true || false;

static P: isize = 3 & 1;
static Q: usize = 1 & 3;

static R: isize = 3 | 1;
static S: usize = 1 | 3;

static T: isize = 3 ^ 1;
static U: usize = 1 ^ 3;

static V: isize = 1 << 3;

// NOTE: better shr coverage
static W: isize = 1024 >> 4;
static X: usize = 1024 >> 4;

static Y: bool = 1 == 1;
static Z: bool = 1.0f64 == 1.0;

static AA: bool = 1 <= 2;
static AB: bool = -1 <= 2;
static AC: bool = 1.0f64 <= 2.0;

static AD: bool = 1 < 2;
static AE: bool = -1 < 2;
static AF: bool = 1.0f64 < 2.0;

static AG: bool = 1 != 2;
static AH: bool = -1 != 2;
static AI: bool = 1.0f64 != 2.0;

static AJ: bool = 2 >= 1;
static AK: bool = 2 >= -2;
static AL: bool = 1.0f64 >= -2.0;

static AM: bool = 2 > 1;
static AN: bool = 2 > -2;
static AO: bool = 1.0f64 > -2.0;

pub fn main() {
    assert_eq!(A, -1);
    assert_eq!(A2, 6);
    assert_approx_eq!(B, 5.7);

    assert_eq!(C, -1);
    assert_eq!(D, 0);
    assert_approx_eq!(E, 0.3);

    assert_eq!(E2, -9);
    assert_eq!(F, 9);
    assert_approx_eq!(G, 10.89);

    assert_eq!(H, -3);
    assert_eq!(I, 1);
    assert_approx_eq!(J, 1.0);

    assert_eq!(N, false);

    assert_eq!(O, true);

    assert_eq!(P, 1);
    assert_eq!(Q, 1);

    assert_eq!(R, 3);
    assert_eq!(S, 3);

    assert_eq!(T, 2);
    assert_eq!(U, 2);

    assert_eq!(V, 8);

    assert_eq!(W, 64);
    assert_eq!(X, 64);

    assert_eq!(Y, true);
    assert_eq!(Z, true);

    assert_eq!(AA, true);
    assert_eq!(AB, true);
    assert_eq!(AC, true);

    assert_eq!(AD, true);
    assert_eq!(AE, true);
    assert_eq!(AF, true);

    assert_eq!(AG, true);
    assert_eq!(AH, true);
    assert_eq!(AI, true);

    assert_eq!(AJ, true);
    assert_eq!(AK, true);
    assert_eq!(AL, true);

    assert_eq!(AM, true);
    assert_eq!(AN, true);
    assert_eq!(AO, true);
}
