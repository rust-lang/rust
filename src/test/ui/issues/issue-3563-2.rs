// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

trait Canvas {
    fn add_point(&self, point: &isize);
    fn add_points(&self, shapes: &[isize]) {
        for pt in shapes {
            self.add_point(pt)
        }
    }

}

pub fn main() {}
