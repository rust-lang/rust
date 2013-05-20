static a: int = -4 + 3;
static a2: uint = 3 + 3;
static b: float = 3.0 + 2.7;

static c: int = 3 - 4;
static d: uint = 3 - 3;
static e: float = 3.0 - 2.7;

static e2: int = -3 * 3;
static f: uint = 3 * 3;
static g: float = 3.3 * 3.3;

static h: int = 3 / -1;
static i: uint = 3 / 3;
static j: float = 3.3 / 3.3;

static n: bool = true && false;

static o: bool = true || false;

static p: int = 3 & 1;
static q: uint = 1 & 3;

static r: int = 3 | 1;
static s: uint = 1 | 3;

static t: int = 3 ^ 1;
static u: uint = 1 ^ 3;

static v: int = 1 << 3;

// NOTE: better shr coverage
static w: int = 1024 >> 4;
static x: uint = 1024 >> 4;

static y: bool = 1 == 1;
static z: bool = 1.0 == 1.0;

static aa: bool = 1 <= 2;
static ab: bool = -1 <= 2;
static ac: bool = 1.0 <= 2.0;

static ad: bool = 1 < 2;
static ae: bool = -1 < 2;
static af: bool = 1.0 < 2.0;

static ag: bool = 1 != 2;
static ah: bool = -1 != 2;
static ai: bool = 1.0 != 2.0;

static aj: bool = 2 >= 1;
static ak: bool = 2 >= -2;
static al: bool = 1.0 >= -2.0;

static am: bool = 2 > 1;
static an: bool = 2 > -2;
static ao: bool = 1.0 > -2.0;

fn main() {
    assert_eq!(a, -1);
    assert_eq!(a2, 6);
    assert_approx_eq!(b, 5.7);

    assert_eq!(c, -1);
    assert_eq!(d, 0);
    assert_approx_eq!(e, 0.3);

    assert_eq!(e2, -9);
    assert_eq!(f, 9);
    assert_approx_eq!(g, 10.89);

    assert_eq!(h, -3);
    assert_eq!(i, 1);
    assert_approx_eq!(j, 1.0);

    assert_eq!(n, false);

    assert_eq!(o, true);

    assert_eq!(p, 1);
    assert_eq!(q, 1);

    assert_eq!(r, 3);
    assert_eq!(s, 3);

    assert_eq!(t, 2);
    assert_eq!(u, 2);

    assert_eq!(v, 8);

    assert_eq!(w, 64);
    assert_eq!(x, 64);

    assert_eq!(y, true);
    assert_eq!(z, true);

    assert_eq!(aa, true);
    assert_eq!(ab, true);
    assert_eq!(ac, true);

    assert_eq!(ad, true);
    assert_eq!(ae, true);
    assert_eq!(af, true);

    assert_eq!(ag, true);
    assert_eq!(ah, true);
    assert_eq!(ai, true);

    assert_eq!(aj, true);
    assert_eq!(ak, true);
    assert_eq!(al, true);

    assert_eq!(am, true);
    assert_eq!(an, true);
    assert_eq!(ao, true);
}
