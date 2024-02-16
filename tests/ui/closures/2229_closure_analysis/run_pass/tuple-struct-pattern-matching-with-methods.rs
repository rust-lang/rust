//@ edition:2021
//@check-pass

#[derive(Copy, Clone)]
enum PointType {
    TwoD(u32, u32),
    ThreeD(u32, u32, u32)
}

// Testing tuple struct patterns
struct Points {
    points: Vec<PointType>,
}

impl Points {
    pub fn test1(&mut self) -> Vec<usize> {
        (0..self.points.len())
            .filter_map(|i| {
                match self.test2(i) {
                    PointType::TwoD (..) => Some(i),
                    PointType::ThreeD (..) => None,
                }
            })
            .collect()
    }

    pub fn test2(&mut self, i: usize) -> PointType {
        self.points[i]
    }
}

fn main() {
    let mut points = Points {
        points: Vec::<PointType>::new()
    };

    points.points.push(PointType::ThreeD(0,0,0));
    points.points.push(PointType::TwoD(0,0));
    points.points.push(PointType::ThreeD(0,0,1));
    points.points.push(PointType::TwoD(0,1));

    println!("{:?}", points.test1());
}
