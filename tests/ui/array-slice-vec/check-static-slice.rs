// run-pass

// Check that the various ways of getting to a reference to a vec (both sized
// and unsized) work properly.


const AA: [isize; 3] = [1, 2, 3];
const AB: &'static [isize; 3] = &AA;
const AC: &'static [isize] = AB;
const AD: &'static [isize] = &AA;
const AE: &'static [isize; 3] = &[1, 2, 3];
const AF: &'static [isize] = &[1, 2, 3];

static CA: isize = AA[0];
static CB: isize = AB[1];
static CC: isize = AC[2];
static CD: isize = AD[0];
static CE: isize = AE[1];
static CF: isize = AF[2];

static AG: &'static isize = &AA[2];

fn main () {
    let b: &[isize] = &[1, 2, 3];
    assert_eq!(AC, b);
    assert_eq!(AD, b);
    assert_eq!(AF, b);
    assert_eq!(*AG, 3);

    assert_eq!(CA, 1);
    assert_eq!(CB, 2);
    assert_eq!(CC, 3);
    assert_eq!(CD, 1);
    assert_eq!(CE, 2);
    assert_eq!(CF, 3);
}
