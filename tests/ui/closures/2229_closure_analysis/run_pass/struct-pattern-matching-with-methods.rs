//@ edition:2021
//@check-pass
#![warn(unused)]
#![allow(dead_code)]
#![feature(rustc_attrs)]

#[derive(Debug, Clone, Copy)]
enum PointType {
    TwoD { x: u32, y: u32 },

    ThreeD{ x: u32, y: u32, z: u32 }
}

// Testing struct patterns
struct Points {
    points: Vec<PointType>,
}

impl Points {
    pub fn test1(&mut self) -> Vec<usize> {
        (0..self.points.len())
            .filter_map(|i| {
                let idx = i as usize;
                match self.test2(idx) {
                    PointType::TwoD { .. } => Some(i),
                    PointType::ThreeD { .. } => None,
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

    points.points.push(PointType::ThreeD { x:0, y:0, z:0 });
    points.points.push(PointType::TwoD{ x:0, y:0 });
    points.points.push(PointType::ThreeD{ x:0, y:0, z:0 });
    points.points.push(PointType::TwoD{ x:0, y:0 });

    println!("{:?}", points.test1());
    println!("{:?}", points.points);
}
