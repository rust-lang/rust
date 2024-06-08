#[doc(alias = "加法")]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub fn 中文名称的加法API(left: usize, right: usize) -> usize {
    left + right
}

#[macro_export]
macro_rules! 中文名称的加法宏 {
    ($left:expr, $right:expr) => {
        ($left) + ($right)
    };
}

#[doc(alias = "加法")]
#[macro_export]
macro_rules! add {
    ($left:expr, $right:expr) => {
        ($left) + ($right)
    };
}

/// Add
pub trait 加法<类型> {
    type 结果;
    fn 加上(self, 被加数: 类型) -> Self::结果;
}

/// IntoIterator
pub trait 可迭代 {
    type 项;
    type 转为迭代器: Iterator<Item = Self::项>;
    fn 迭代(self) -> Self::转为迭代器;
}

pub type 可选<类型> = Option<类型>;

/// "sum"
pub fn 总计<集合, 个体>(容器: 集合) -> 可选<集合::项>
where
    集合: 可迭代<项 = 个体>,
    个体: 加法<个体, 结果 = 个体>,
{
    容器.迭代().reduce(|累计值, 当前值| 累计值.加上(当前值))
}
