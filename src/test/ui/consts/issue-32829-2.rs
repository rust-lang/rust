const bad : u32 = {
    {
        5;
        0
    }
};

const bad_two : u32 = {
    {
        invalid();
        //~^ ERROR: cannot call non-const fn `invalid`
        0
    }
};

const bad_three : u32 = {
    {
        valid();
        0
    }
};

static bad_four : u32 = {
    {
        5;
        0
    }
};

static bad_five : u32 = {
    {
        invalid();
        //~^ ERROR: cannot call non-const fn `invalid`
        0
    }
};

static bad_six : u32 = {
    {
        valid();
        0
    }
};

static mut bad_seven : u32 = {
    {
        5;
        0
    }
};

static mut bad_eight : u32 = {
    {
        invalid();
        //~^ ERROR: cannot call non-const fn `invalid`
        0
    }
};

static mut bad_nine : u32 = {
    {
        valid();
        0
    }
};


fn invalid() {}
const fn valid() {}

fn main() {}
