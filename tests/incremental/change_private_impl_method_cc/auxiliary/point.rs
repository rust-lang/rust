pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    fn distance_squared(&self) -> f32 {
        #[cfg(cfail1)]
        return self.x + self.y;

        #[cfg(cfail2)]
        return self.x * self.x + self.y * self.y;
    }

    pub fn distance_from_origin(&self) -> f32 {
        self.distance_squared().sqrt()
    }
}

impl Point {
    pub fn translate(&mut self, x: f32, y: f32) {
        self.x += x;
        self.y += y;
    }
}
